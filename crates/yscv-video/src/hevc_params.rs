//! HEVC parameter set structs and parsing (VPS, SPS, PPS, slice header).

use super::h264_bitstream::BitstreamReader;
use crate::VideoError;

// ---------------------------------------------------------------------------
// Video Parameter Set (VPS)
// ---------------------------------------------------------------------------

/// HEVC Video Parameter Set.
#[derive(Debug, Clone)]
pub struct HevcVps {
    pub vps_id: u8,
    pub max_layers: u8,
    pub max_sub_layers: u8,
    pub temporal_id_nesting: bool,
}

// ---------------------------------------------------------------------------
// Sequence Parameter Set (SPS)
// ---------------------------------------------------------------------------

/// HEVC Sequence Parameter Set.
#[derive(Debug, Clone)]
pub struct HevcSps {
    pub sps_id: u8,
    pub vps_id: u8,
    pub max_sub_layers: u8,
    pub chroma_format_idc: u8, // 0=mono, 1=4:2:0, 2=4:2:2, 3=4:4:4
    pub pic_width: u32,
    pub pic_height: u32,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_pic_order_cnt: u8,
    pub log2_min_cb_size: u8,
    pub log2_diff_max_min_cb_size: u8,
    pub log2_min_transform_size: u8,
    pub log2_diff_max_min_transform_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub sample_adaptive_offset_enabled: bool,
    pub pcm_enabled: bool,
    pub num_short_term_ref_pic_sets: u8,
    pub long_term_ref_pics_present: bool,
    pub sps_temporal_mvp_enabled: bool,
    pub strong_intra_smoothing_enabled: bool,
}

// ---------------------------------------------------------------------------
// Picture Parameter Set (PPS)
// ---------------------------------------------------------------------------

/// HEVC Picture Parameter Set.
#[derive(Debug, Clone)]
pub struct HevcPps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default: u8,
    pub num_ref_idx_l1_default: u8,
    pub init_qp: i8,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub cu_qp_delta_enabled: bool,
    pub cb_qp_offset: i8,
    pub cr_qp_offset: i8,
    pub deblocking_filter_override_enabled: bool,
    pub deblocking_filter_disabled: bool,
    pub loop_filter_across_slices_enabled: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    /// Number of tile columns (1 = no tiles).
    pub num_tile_columns: u32,
    /// Number of tile rows (1 = no tiles).
    pub num_tile_rows: u32,
    /// Column boundaries in CTU units (length = num_tile_columns + 1).
    pub tile_col_boundaries: Vec<u32>,
    /// Row boundaries in CTU units (length = num_tile_rows + 1).
    pub tile_row_boundaries: Vec<u32>,
    /// Whether loop filter crosses tile boundaries.
    pub loop_filter_across_tiles_enabled: bool,
}

// ---------------------------------------------------------------------------
// Slice header & slice types
// ---------------------------------------------------------------------------

/// HEVC weighted prediction entry for one reference.
#[derive(Debug, Clone, Default)]
pub struct HevcWeightEntry {
    pub weight: i32,
    pub offset: i32,
}

/// HEVC weighted prediction table (pred_weight_table §7.3.6.3).
#[derive(Debug, Clone, Default)]
pub struct HevcWeightTable {
    pub luma_log2_denom: u32,
    pub chroma_log2_denom: u32,
    pub luma_l0: Vec<HevcWeightEntry>,
    pub luma_l1: Vec<HevcWeightEntry>,
}

/// Parse HEVC pred_weight_table from a bitstream reader (ITU-T H.265 §7.3.6.3).
pub fn parse_hevc_weight_table(
    reader: &mut super::h264_bitstream::BitstreamReader<'_>,
    slice_type: HevcSliceType,
    num_ref_l0: u32,
    num_ref_l1: u32,
) -> Result<HevcWeightTable, super::VideoError> {
    let luma_log2_denom = reader.read_ue()?;
    let chroma_log2_denom =
        luma_log2_denom.wrapping_add_signed(reader.read_se()? as i64 as isize as i32 as i64 as i32);

    let luma_default = 1i32 << luma_log2_denom;
    let mut luma_l0 = Vec::new();
    // luma_weight_l0_flag for each ref
    let mut luma_flags = Vec::new();
    for _ in 0..num_ref_l0 {
        luma_flags.push(reader.read_bit()? != 0);
    }
    for i in 0..num_ref_l0 as usize {
        if luma_flags[i] {
            let w = reader.read_se()? + luma_default;
            let o = reader.read_se()?;
            luma_l0.push(HevcWeightEntry {
                weight: w,
                offset: o,
            });
        } else {
            luma_l0.push(HevcWeightEntry {
                weight: luma_default,
                offset: 0,
            });
        }
    }

    let mut luma_l1 = Vec::new();
    if slice_type == HevcSliceType::B {
        let mut l1_flags = Vec::new();
        for _ in 0..num_ref_l1 {
            l1_flags.push(reader.read_bit()? != 0);
        }
        for i in 0..num_ref_l1 as usize {
            if l1_flags[i] {
                let w = reader.read_se()? + luma_default;
                let o = reader.read_se()?;
                luma_l1.push(HevcWeightEntry {
                    weight: w,
                    offset: o,
                });
            } else {
                luma_l1.push(HevcWeightEntry {
                    weight: luma_default,
                    offset: 0,
                });
            }
        }
    }

    Ok(HevcWeightTable {
        luma_log2_denom,
        chroma_log2_denom,
        luma_l0,
        luma_l1,
    })
}

/// HEVC Slice Header (simplified).
#[derive(Debug, Clone)]
pub struct HevcSliceHeader {
    pub first_slice_in_pic: bool,
    pub slice_type: HevcSliceType,
    pub pps_id: u8,
    pub slice_qp_delta: i8,
    pub weight_table: Option<HevcWeightTable>,
}

/// HEVC slice types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcSliceType {
    B = 0,
    P = 1,
    I = 2,
}

// ---------------------------------------------------------------------------
// VPS parsing
// ---------------------------------------------------------------------------

/// Parse HEVC VPS from NAL unit payload (after the 2-byte NAL header).
pub fn parse_hevc_vps(data: &[u8]) -> Result<HevcVps, VideoError> {
    let rbsp = super::h264_params::remove_emulation_prevention(data);
    let mut reader = BitstreamReader::new(&rbsp);
    let vps_id = reader.read_bits(4)? as u8;
    reader.read_bits(2)?; // vps_base_layer_internal_flag + vps_base_layer_available_flag
    let max_layers = reader.read_bits(6)? as u8 + 1;
    let max_sub_layers = reader.read_bits(3)? as u8 + 1;
    let temporal_id_nesting = reader.read_bit()? != 0;
    Ok(HevcVps {
        vps_id,
        max_layers,
        max_sub_layers,
        temporal_id_nesting,
    })
}

// ---------------------------------------------------------------------------
// Profile-tier-level skipping
// ---------------------------------------------------------------------------

/// Skip profile_tier_level syntax element.
fn skip_profile_tier_level(
    reader: &mut BitstreamReader,
    max_sub_layers: u8,
) -> Result<(), VideoError> {
    // general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5) = 8 bits
    reader.read_bits(8)?;
    // general_profile_compatibility_flags (32 bits)
    reader.read_bits(16)?;
    reader.read_bits(16)?;
    // general_constraint_indicator_flags (48 bits)
    reader.read_bits(16)?;
    reader.read_bits(16)?;
    reader.read_bits(16)?;
    // general_level_idc (8 bits)
    reader.read_bits(8)?;
    // sub_layer flags (if max_sub_layers > 1)
    for _ in 1..max_sub_layers {
        reader.read_bits(2)?; // sub_layer_profile_present + sub_layer_level_present
    }
    if max_sub_layers > 1 {
        for _ in max_sub_layers..8 {
            reader.read_bits(2)?; // reserved zero 2 bits
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SPS parsing
// ---------------------------------------------------------------------------

/// Parse HEVC SPS from NAL unit payload (after the 2-byte NAL header).
pub fn parse_hevc_sps(data: &[u8]) -> Result<HevcSps, VideoError> {
    let rbsp = super::h264_params::remove_emulation_prevention(data);
    let mut reader = BitstreamReader::new(&rbsp);
    let vps_id = reader.read_bits(4)? as u8;
    let max_sub_layers = reader.read_bits(3)? as u8 + 1;
    let _temporal_id_nesting = reader.read_bit()?;

    // Skip profile_tier_level (simplified -- fixed-length approximation)
    skip_profile_tier_level(&mut reader, max_sub_layers)?;

    let sps_id = reader.read_ue()? as u8;
    let chroma_format_idc = reader.read_ue()? as u8;
    if chroma_format_idc == 3 {
        reader.read_bit()?; // separate_colour_plane_flag
    }
    let pic_width = reader.read_ue()?;
    let pic_height = reader.read_ue()?;

    let conformance_window = reader.read_bit()? != 0;
    if conformance_window {
        reader.read_ue()?; // conf_win_left_offset
        reader.read_ue()?; // conf_win_right_offset
        reader.read_ue()?; // conf_win_top_offset
        reader.read_ue()?; // conf_win_bottom_offset
    }

    let bit_depth_luma = reader.read_ue()? as u8 + 8;
    let bit_depth_chroma = reader.read_ue()? as u8 + 8;
    let log2_max_pic_order_cnt = reader.read_ue()? as u8 + 4;

    // sub_layer_ordering_info_present_flag
    let sub_layer_ordering_info_present = reader.read_bit()? != 0;
    let start = if sub_layer_ordering_info_present {
        0
    } else {
        max_sub_layers - 1
    };
    for _ in start..max_sub_layers {
        reader.read_ue()?; // max_dec_pic_buffering_minus1
        reader.read_ue()?; // max_num_reorder_pics
        reader.read_ue()?; // max_latency_increase_plus1
    }

    let log2_min_cb_size = reader.read_ue()? as u8 + 3;
    let log2_diff_max_min_cb_size = reader.read_ue()? as u8;
    let log2_min_transform_size = reader.read_ue()? as u8 + 2;
    let log2_diff_max_min_transform_size = reader.read_ue()? as u8;
    let max_transform_hierarchy_depth_inter = reader.read_ue()? as u8;
    let max_transform_hierarchy_depth_intra = reader.read_ue()? as u8;

    // scaling_list_enabled_flag
    let scaling_list_enabled = reader.read_bit()? != 0;
    if scaling_list_enabled {
        let scaling_list_data_present = reader.read_bit()? != 0;
        if scaling_list_data_present {
            skip_scaling_list_data(&mut reader)?;
        }
    }

    // amp_enabled_flag, sample_adaptive_offset_enabled_flag
    let _amp_enabled = reader.read_bit()?;
    let sample_adaptive_offset_enabled = reader.read_bit()? != 0;

    // pcm_enabled_flag
    let pcm_enabled = reader.read_bit()? != 0;
    if pcm_enabled {
        // pcm_sample_bit_depth_luma_minus1 (4) + pcm_sample_bit_depth_chroma_minus1 (4)
        reader.read_bits(4)?;
        reader.read_bits(4)?;
        reader.read_ue()?; // log2_min_pcm_luma_coding_block_size_minus3
        reader.read_ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
        reader.read_bit()?; // pcm_loop_filter_disabled_flag
    }

    let num_short_term_ref_pic_sets = reader.read_ue()? as u8;
    // Skip actual short-term ref pic set parsing (complex; fill defaults below)

    // For remaining flags that require parsing the ref pic sets first,
    // use conservative defaults.
    Ok(HevcSps {
        sps_id,
        vps_id,
        max_sub_layers,
        chroma_format_idc,
        pic_width,
        pic_height,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_pic_order_cnt,
        log2_min_cb_size,
        log2_diff_max_min_cb_size,
        log2_min_transform_size,
        log2_diff_max_min_transform_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        sample_adaptive_offset_enabled,
        pcm_enabled,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present: false,
        sps_temporal_mvp_enabled: false,
        strong_intra_smoothing_enabled: false,
    })
}

// ---------------------------------------------------------------------------
// PPS parsing
// ---------------------------------------------------------------------------

/// Parse HEVC PPS from NAL unit payload (after the 2-byte NAL header).
pub fn parse_hevc_pps(data: &[u8]) -> Result<HevcPps, VideoError> {
    let rbsp = super::h264_params::remove_emulation_prevention(data);
    let mut reader = BitstreamReader::new(&rbsp);

    let pps_id = reader.read_ue()? as u8;
    let sps_id = reader.read_ue()? as u8;
    let dependent_slice_segments_enabled = reader.read_bit()? != 0;
    let output_flag_present = reader.read_bit()? != 0;
    let num_extra_slice_header_bits = reader.read_bits(3)? as u8;
    let sign_data_hiding_enabled = reader.read_bit()? != 0;
    let cabac_init_present = reader.read_bit()? != 0;
    let num_ref_idx_l0_default = reader.read_ue()? as u8 + 1;
    let num_ref_idx_l1_default = reader.read_ue()? as u8 + 1;
    let init_qp_minus26 = reader.read_se()?;
    let init_qp = (26 + init_qp_minus26) as i8;
    let constrained_intra_pred = reader.read_bit()? != 0;
    let transform_skip_enabled = reader.read_bit()? != 0;
    let cu_qp_delta_enabled = reader.read_bit()? != 0;
    if cu_qp_delta_enabled {
        reader.read_ue()?; // diff_cu_qp_delta_depth
    }
    let cb_qp_offset = reader.read_se()? as i8;
    let cr_qp_offset = reader.read_se()? as i8;
    let _slice_chroma_qp_offsets_present = reader.read_bit()?;
    let weighted_pred_flag = reader.read_bit()? != 0;
    let weighted_bipred_flag = reader.read_bit()? != 0;
    let _transquant_bypass_enabled = reader.read_bit()?;
    let tiles_enabled = reader.read_bit()? != 0;
    let entropy_coding_sync_enabled = reader.read_bit()? != 0;

    let mut num_tile_columns = 1u32;
    let mut num_tile_rows = 1u32;
    let mut tile_col_boundaries = Vec::new();
    let mut tile_row_boundaries = Vec::new();
    let mut loop_filter_across_tiles_enabled = true;

    if tiles_enabled {
        num_tile_columns = reader.read_ue()? + 1;
        num_tile_rows = reader.read_ue()? + 1;
        let uniform_spacing = reader.read_bit()? != 0;
        if !uniform_spacing {
            // Read explicit column widths in CTU units
            for _ in 0..num_tile_columns - 1 {
                tile_col_boundaries.push(reader.read_ue()? + 1);
            }
            for _ in 0..num_tile_rows - 1 {
                tile_row_boundaries.push(reader.read_ue()? + 1);
            }
        }
        if tiles_enabled || entropy_coding_sync_enabled {
            loop_filter_across_tiles_enabled = reader.read_bit()? != 0;
        }
    }

    let loop_filter_across_slices_enabled = reader.read_bit()? != 0;
    let deblocking_filter_control_present = reader.read_bit()? != 0;
    let mut deblocking_filter_override_enabled = false;
    let mut deblocking_filter_disabled = false;
    if deblocking_filter_control_present {
        deblocking_filter_override_enabled = reader.read_bit()? != 0;
        deblocking_filter_disabled = reader.read_bit()? != 0;
        if !deblocking_filter_disabled {
            reader.read_se()?; // pps_beta_offset_div2
            reader.read_se()?; // pps_tc_offset_div2
        }
    }

    Ok(HevcPps {
        pps_id,
        sps_id,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        cabac_init_present,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        init_qp,
        constrained_intra_pred,
        transform_skip_enabled,
        cu_qp_delta_enabled,
        cb_qp_offset,
        cr_qp_offset,
        deblocking_filter_override_enabled,
        deblocking_filter_disabled,
        loop_filter_across_slices_enabled,
        weighted_pred_flag,
        weighted_bipred_flag,
        tiles_enabled,
        entropy_coding_sync_enabled,
        num_tile_columns,
        num_tile_rows,
        tile_col_boundaries,
        tile_row_boundaries,
        loop_filter_across_tiles_enabled,
    })
}

// ---------------------------------------------------------------------------
// Scaling list data parsing (S7.3.4)
// ---------------------------------------------------------------------------

/// Parse and discard scaling_list_data() per HEVC spec S7.3.4.
/// Advances the bitstream position correctly without storing values.
pub(crate) fn skip_scaling_list_data(reader: &mut BitstreamReader) -> Result<(), VideoError> {
    for size_id in 0..4u8 {
        let matrix_count: u8 = if size_id == 3 { 2 } else { 6 };
        let matrix_step: u8 = if size_id == 3 { 3 } else { 1 };
        for matrix_idx in 0..matrix_count {
            let _matrix_id = matrix_idx * matrix_step;
            let pred_mode_flag = reader.read_bit()?;
            if pred_mode_flag == 0 {
                // scaling_list_pred_matrix_id_delta
                reader.read_ue()?;
            } else {
                let coef_num = std::cmp::min(64, 1u32 << (4 + (u32::from(size_id) << 1)));
                if size_id > 1 {
                    // scaling_list_dc_coef_minus8
                    reader.read_se()?;
                }
                for _ in 0..coef_num {
                    // scaling_list_delta_coef
                    reader.read_se()?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract frame dimensions from HEVC SPS.
pub fn hevc_frame_dimensions(sps: &HevcSps) -> (u32, u32) {
    (sps.pic_width, sps.pic_height)
}
