//! H.264 parameter set parsing (SPS, PPS, slice header).

use super::h264_bitstream::BitstreamReader;
use crate::VideoError;

// ---------------------------------------------------------------------------
// SPS parsing
// ---------------------------------------------------------------------------

/// Parsed Sequence Parameter Set (subset of fields needed for frame dimensions).
#[derive(Debug, Clone)]
pub struct Sps {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub sps_id: u32,
    pub chroma_format_idc: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_frame_num: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb: u32,
    pub max_num_ref_frames: u32,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub frame_crop_left: u32,
    pub frame_crop_right: u32,
    pub frame_crop_top: u32,
    pub frame_crop_bottom: u32,
    /// 4x4 scaling lists (6 lists of 16 values). Default = flat 16.
    pub scaling_list_4x4: [[i32; 16]; 6],
    /// 8x8 scaling lists (6 lists of 64 values). Default = flat 16.
    pub scaling_list_8x8: [[i32; 64]; 6],
}

impl Sps {
    /// Frame width in pixels (before cropping).
    pub fn width(&self) -> usize {
        self.pic_width_in_mbs
            .checked_mul(16)
            .map(|v| v as usize)
            .unwrap_or(usize::MAX)
    }

    /// Frame height in pixels (before cropping).
    pub fn height(&self) -> usize {
        let mbs_height = if self.frame_mbs_only_flag {
            self.pic_height_in_map_units
        } else {
            self.pic_height_in_map_units.saturating_mul(2)
        };
        mbs_height
            .checked_mul(16)
            .map(|v| v as usize)
            .unwrap_or(usize::MAX)
    }

    /// Cropped frame width.
    ///
    /// Returns the full width if cropping would underflow (malformed SPS).
    pub fn cropped_width(&self) -> usize {
        let sub_width_c = if self.chroma_format_idc == 1 { 2 } else { 1 };
        let crop = (self.frame_crop_left + self.frame_crop_right) as usize * sub_width_c;
        self.width().saturating_sub(crop).max(1)
    }

    /// Cropped frame height.
    ///
    /// Returns the full height if cropping would underflow (malformed SPS).
    pub fn cropped_height(&self) -> usize {
        let sub_height_c = if self.chroma_format_idc == 1 { 2 } else { 1 };
        let factor = if self.frame_mbs_only_flag { 1 } else { 2 };
        let crop = (self.frame_crop_top + self.frame_crop_bottom) as usize * sub_height_c * factor;
        self.height().saturating_sub(crop).max(1)
    }
}

/// Parses an SPS NAL unit (without the NAL header byte).
pub fn parse_sps(nal_data: &[u8]) -> Result<Sps, VideoError> {
    if nal_data.is_empty() {
        return Err(VideoError::Codec("empty SPS data".into()));
    }

    // Remove emulation prevention bytes (0x00 0x00 0x03 -> 0x00 0x00)
    let rbsp = remove_emulation_prevention(nal_data);
    let mut r = BitstreamReader::new(&rbsp);

    let profile_idc = r.read_bits(8)? as u8;
    let _constraint_flags = r.read_bits(8)?; // constraint_set0..5_flag + reserved
    let level_idc = r.read_bits(8)? as u8;
    let sps_id = r.read_ue()?;

    let mut chroma_format_idc = 1u32;
    let mut bit_depth_luma = 8u32;
    let mut scaling_list_4x4 = [[16i32; 16]; 6];
    let mut scaling_list_8x8 = [[16i32; 64]; 6];
    let mut bit_depth_chroma = 8u32;

    // High profile extensions
    if profile_idc == 100
        || profile_idc == 110
        || profile_idc == 122
        || profile_idc == 244
        || profile_idc == 44
        || profile_idc == 83
        || profile_idc == 86
        || profile_idc == 118
        || profile_idc == 128
    {
        chroma_format_idc = r.read_ue()?;
        if chroma_format_idc == 3 {
            let _separate_colour_plane_flag = r.read_bit()?;
        }
        bit_depth_luma = r.read_ue()? + 8;
        bit_depth_chroma = r.read_ue()? + 8;
        let _qpprime_y_zero_transform_bypass = r.read_bit()?;
        let seq_scaling_matrix_present = r.read_bit()?;
        if seq_scaling_matrix_present == 1 {
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let present = r.read_bit()?;
                if present == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    let list = parse_scaling_list(&mut r, size)?;
                    if i < 6 && list.len() == 16 {
                        scaling_list_4x4[i].copy_from_slice(&list);
                    } else if i >= 6 && list.len() == 64 {
                        scaling_list_8x8[i - 6].copy_from_slice(&list);
                    }
                }
            }
        }
    }

    let log2_max_frame_num = r.read_ue()? + 4;
    let pic_order_cnt_type = r.read_ue()?;

    let mut log2_max_pic_order_cnt_lsb = 0u32;
    if pic_order_cnt_type == 0 {
        log2_max_pic_order_cnt_lsb = r.read_ue()? + 4;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = r.read_bit()?;
        let _offset_for_non_ref_pic = r.read_se()?;
        let _offset_for_top_to_bottom = r.read_se()?;
        let num_ref_frames_in_poc = r.read_ue()?;
        if num_ref_frames_in_poc > 255 {
            return Err(VideoError::Codec(format!(
                "SPS num_ref_frames_in_pic_order_cnt_cycle too large: {num_ref_frames_in_poc}"
            )));
        }
        for _ in 0..num_ref_frames_in_poc {
            let _offset = r.read_se()?;
        }
    }

    let max_num_ref_frames = r.read_ue()?;
    let _gaps_in_frame_num_allowed = r.read_bit()?;
    let pic_width_in_mbs = r.read_ue()? + 1;
    let pic_height_in_map_units = r.read_ue()? + 1;
    let frame_mbs_only_flag = r.read_bit()? == 1;

    let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
        r.read_bit()? == 1
    } else {
        false
    };

    let _direct_8x8_inference = r.read_bit()?;

    let mut frame_crop_left = 0u32;
    let mut frame_crop_right = 0u32;
    let mut frame_crop_top = 0u32;
    let mut frame_crop_bottom = 0u32;

    let frame_cropping_flag = r.read_bit()?;
    if frame_cropping_flag == 1 {
        frame_crop_left = r.read_ue()?;
        frame_crop_right = r.read_ue()?;
        frame_crop_top = r.read_ue()?;
        frame_crop_bottom = r.read_ue()?;
    }

    Ok(Sps {
        profile_idc,
        level_idc,
        sps_id,
        chroma_format_idc,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_frame_num,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb,
        max_num_ref_frames,
        pic_width_in_mbs,
        pic_height_in_map_units,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        frame_crop_left,
        frame_crop_right,
        frame_crop_top,
        frame_crop_bottom,
        scaling_list_4x4,
        scaling_list_8x8,
    })
}

/// Parse a scaling list from the bitstream and return it.
/// If `use_default` is true (seq_scaling_list_present_flag=0), returns None (use default flat).
fn parse_scaling_list(r: &mut BitstreamReader<'_>, size: usize) -> Result<Vec<i32>, VideoError> {
    let mut list = vec![16i32; size]; // default flat
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for j in 0..size {
        if next_scale != 0 {
            let delta = r.read_se()?;
            next_scale = (last_scale + delta + 256) % 256;
        }
        let scale = if next_scale == 0 {
            last_scale
        } else {
            next_scale
        };
        list[j] = scale;
        last_scale = scale;
    }
    Ok(list)
}

/// Removes H.264 emulation prevention bytes (0x00 0x00 0x03 -> 0x00 0x00).
pub(crate) fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            result.push(0x00);
            result.push(0x00);
            i += 3; // skip the 0x03
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// PPS parsing
// ---------------------------------------------------------------------------

/// Parsed Picture Parameter Set (subset).
#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub num_slice_groups: u32,
    pub slice_group_map_type: u32,
    /// Run-length values for FMO type 0 (interleaved).
    pub run_length_minus1: Vec<u32>,
    /// Top-left MB indices for FMO type 2 (foreground regions).
    pub top_left: Vec<u32>,
    /// Bottom-right MB indices for FMO type 2 (foreground regions).
    pub bottom_right: Vec<u32>,
    /// For FMO types 3-5: direction of changing slice groups.
    pub slice_group_change_direction_flag: bool,
    /// For FMO types 3-5: rate of change in MBs.
    pub slice_group_change_rate: u32,
    /// Explicit MB-to-slice-group map for FMO type 6.
    pub slice_group_id: Vec<u32>,
    pub num_ref_idx_l0_default_active: u32,
    pub num_ref_idx_l1_default_active: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
}

/// Parses a PPS NAL unit (without the NAL header byte).
pub fn parse_pps(nal_data: &[u8]) -> Result<Pps, VideoError> {
    if nal_data.is_empty() {
        return Err(VideoError::Codec("empty PPS data".into()));
    }
    let rbsp = remove_emulation_prevention(nal_data);
    let mut r = BitstreamReader::new(&rbsp);

    let pps_id = r.read_ue()?;
    let sps_id = r.read_ue()?;
    let entropy_coding_mode_flag = r.read_bit()? == 1;
    let _bottom_field_pic_order = r.read_bit()?;
    let num_slice_groups = r.read_ue()? + 1;

    let mut slice_group_map_type = 0u32;
    let mut run_length_minus1 = Vec::new();
    let mut top_left = Vec::new();
    let mut bottom_right = Vec::new();
    let mut slice_group_change_direction_flag = false;
    let mut slice_group_change_rate = 0u32;
    let mut slice_group_id = Vec::new();

    if num_slice_groups > 1 {
        slice_group_map_type = r.read_ue()?;
        match slice_group_map_type {
            0 => {
                // Interleaved: run_length_minus1 for each slice group
                for _ in 0..num_slice_groups {
                    run_length_minus1.push(r.read_ue()?);
                }
            }
            2 => {
                // Foreground with left-over: top_left and bottom_right for each group except last
                for _ in 0..num_slice_groups.saturating_sub(1) {
                    top_left.push(r.read_ue()?);
                    bottom_right.push(r.read_ue()?);
                }
            }
            3..=5 => {
                slice_group_change_direction_flag = r.read_bit()? == 1;
                slice_group_change_rate = r.read_ue()? + 1;
            }
            6 => {
                let pic_size_in_map_units = r.read_ue()? + 1;
                let bits_needed = if num_slice_groups > 1 {
                    (32 - (num_slice_groups - 1).leading_zeros()).max(1) as u8
                } else {
                    1
                };
                for _ in 0..pic_size_in_map_units {
                    slice_group_id.push(r.read_bits(bits_needed)?);
                }
            }
            _ => {
                // Type 1 (dispersed): no additional data needed
            }
        }
    }

    let num_ref_idx_l0_default_active = r.read_ue()? + 1;
    let num_ref_idx_l1_default_active = r.read_ue()? + 1;
    let weighted_pred_flag = r.read_bit()? == 1;
    let weighted_bipred_idc = r.read_bits(2)?;
    // pic_init_qp_minus26
    let pic_init_qp_minus26 = r.read_se()?;
    let pic_init_qp = 26 + pic_init_qp_minus26;
    // pic_init_qs_minus26 (unused, for SP/SI slices)
    let _pic_init_qs = r.read_se()?;
    // chroma_qp_index_offset
    let _chroma_qp_index_offset = r.read_se()?;
    let deblocking_filter_control_present_flag = r.read_bit()? == 1;
    let _constrained_intra_pred_flag = r.read_bit()?;
    let _redundant_pic_cnt_present_flag = r.read_bit()?;

    // High profile PPS extension: transform_8x8_mode_flag + second_chroma_qp_index_offset
    let transform_8x8_mode_flag = if r.bits_remaining() >= 1 {
        r.read_bit().unwrap_or(0) == 1
    } else {
        false
    };
    // Skip remaining high profile fields (scaling_list_8x8, second_chroma_qp_index_offset)
    // if present — we don't need them for basic decode

    Ok(Pps {
        pps_id,
        sps_id,
        entropy_coding_mode_flag,
        num_slice_groups,
        slice_group_map_type,
        run_length_minus1,
        top_left,
        bottom_right,
        slice_group_change_direction_flag,
        slice_group_change_rate,
        slice_group_id,
        num_ref_idx_l0_default_active,
        num_ref_idx_l1_default_active,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp,
        deblocking_filter_control_present_flag,
        transform_8x8_mode_flag,
    })
}

// ---------------------------------------------------------------------------
// Slice header
// ---------------------------------------------------------------------------

/// Weighted prediction parameters for one reference picture + component.
#[derive(Debug, Clone, Default)]
pub struct WeightEntry {
    /// Weight value. Default = (1 << log2_weight_denom).
    pub weight: i32,
    /// Offset value. Default = 0.
    pub offset: i32,
}

/// Weighted prediction table parsed from the slice header.
#[derive(Debug, Clone, Default)]
pub struct WeightTable {
    pub luma_log2_denom: u32,
    pub chroma_log2_denom: u32,
    /// Per-reference luma weights for L0 list.
    pub luma_l0: Vec<WeightEntry>,
    /// Per-reference chroma weights for L0 list (Cb, Cr per ref).
    pub chroma_l0: Vec<[WeightEntry; 2]>,
    /// Per-reference luma weights for L1 list (B-slices).
    pub luma_l1: Vec<WeightEntry>,
    /// Per-reference chroma weights for L1 list (Cb, Cr per ref).
    pub chroma_l1: Vec<[WeightEntry; 2]>,
}

/// Parsed slice header (subset of fields needed for IDR decoding).
#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: u32,
    pub pps_id: u32,
    pub frame_num: u32,
    /// True when this slice is a single field (top or bottom) of an interlaced picture.
    pub field_pic_flag: bool,
    /// When `field_pic_flag` is true, indicates this is the bottom field.
    pub bottom_field_flag: bool,
    pub qp: i32,
    /// Weighted prediction table (populated when PPS weighted_pred_flag is set).
    pub weight_table: Option<WeightTable>,
}

/// Parses the pred_weight_table() from the slice header (H.264 7.3.3.2).
fn parse_weight_table(
    r: &mut BitstreamReader<'_>,
    slice_type: u32,
    num_ref_l0: u32,
    num_ref_l1: u32,
    chroma_format_idc: u32,
) -> Result<WeightTable, VideoError> {
    let luma_log2_denom = r.read_ue()?;
    let chroma_log2_denom = if chroma_format_idc > 0 {
        r.read_ue()?
    } else {
        0
    };

    let luma_default = 1i32 << luma_log2_denom;
    let chroma_default = 1i32 << chroma_log2_denom;

    let mut luma_l0 = Vec::new();
    let mut chroma_l0 = Vec::new();
    for _ in 0..num_ref_l0 {
        let luma_weight_flag = r.read_bit()? == 1;
        if luma_weight_flag {
            let w = r.read_se()?;
            let o = r.read_se()?;
            luma_l0.push(WeightEntry {
                weight: w,
                offset: o,
            });
        } else {
            luma_l0.push(WeightEntry {
                weight: luma_default,
                offset: 0,
            });
        }
        if chroma_format_idc > 0 {
            let chroma_weight_flag = r.read_bit()? == 1;
            if chroma_weight_flag {
                let w_cb = r.read_se()?;
                let o_cb = r.read_se()?;
                let w_cr = r.read_se()?;
                let o_cr = r.read_se()?;
                chroma_l0.push([
                    WeightEntry {
                        weight: w_cb,
                        offset: o_cb,
                    },
                    WeightEntry {
                        weight: w_cr,
                        offset: o_cr,
                    },
                ]);
            } else {
                chroma_l0.push([
                    WeightEntry {
                        weight: chroma_default,
                        offset: 0,
                    },
                    WeightEntry {
                        weight: chroma_default,
                        offset: 0,
                    },
                ]);
            }
        }
    }

    let mut luma_l1 = Vec::new();
    let mut chroma_l1 = Vec::new();
    // B-slices: slice_type == 1 or 6
    if slice_type == 1 || slice_type == 6 {
        for _ in 0..num_ref_l1 {
            let luma_weight_flag = r.read_bit()? == 1;
            if luma_weight_flag {
                let w = r.read_se()?;
                let o = r.read_se()?;
                luma_l1.push(WeightEntry {
                    weight: w,
                    offset: o,
                });
            } else {
                luma_l1.push(WeightEntry {
                    weight: luma_default,
                    offset: 0,
                });
            }
            if chroma_format_idc > 0 {
                let chroma_weight_flag = r.read_bit()? == 1;
                if chroma_weight_flag {
                    let w_cb = r.read_se()?;
                    let o_cb = r.read_se()?;
                    let w_cr = r.read_se()?;
                    let o_cr = r.read_se()?;
                    chroma_l1.push([
                        WeightEntry {
                            weight: w_cb,
                            offset: o_cb,
                        },
                        WeightEntry {
                            weight: w_cr,
                            offset: o_cr,
                        },
                    ]);
                } else {
                    chroma_l1.push([
                        WeightEntry {
                            weight: chroma_default,
                            offset: 0,
                        },
                        WeightEntry {
                            weight: chroma_default,
                            offset: 0,
                        },
                    ]);
                }
            }
        }
    }

    Ok(WeightTable {
        luma_log2_denom,
        chroma_log2_denom,
        luma_l0,
        chroma_l0,
        luma_l1,
        chroma_l1,
    })
}

/// Parses a slice header from RBSP data (after the NAL header byte).
pub(crate) fn parse_slice_header(
    r: &mut BitstreamReader<'_>,
    sps: &Sps,
    pps: &Pps,
    is_idr: bool,
) -> Result<SliceHeader, VideoError> {
    let first_mb_in_slice = r.read_ue()?;
    let slice_type = r.read_ue()?;
    let pps_id = r.read_ue()?;
    let frame_num = r.read_bits(sps.log2_max_frame_num as u8)?;

    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = r.read_bit()? == 1;
        if field_pic_flag {
            bottom_field_flag = r.read_bit()? == 1;
        }
    }

    if is_idr {
        let _idr_pic_id = r.read_ue()?;
    }

    if sps.pic_order_cnt_type == 0 {
        let _pic_order_cnt_lsb = r.read_bits(sps.log2_max_pic_order_cnt_lsb as u8)?;
    }

    let is_i_slice = slice_type == 2 || slice_type == 7;
    let is_p_slice = slice_type == 0 || slice_type == 5;
    let is_b_slice = slice_type == 1 || slice_type == 6;

    // num_ref_idx_active_override_flag + num_ref_idx overrides (for P/B slices)
    if !is_i_slice {
        let num_ref_override = r.read_bit()? == 1;
        if num_ref_override {
            let _num_ref_idx_l0 = r.read_ue()? + 1;
            if is_b_slice {
                let _num_ref_idx_l1 = r.read_ue()? + 1;
            }
        }
    }

    // ref_pic_list_modification() — skip for P/B slices
    if !is_i_slice {
        // ref_pic_list_modification_flag_l0
        let mod_flag_l0 = r.read_bit()? == 1;
        if mod_flag_l0 {
            loop {
                let op = r.read_ue()?;
                if op == 3 {
                    break;
                } // end
                let _val = r.read_ue()?;
                if r.bits_remaining() < 4 {
                    break;
                }
            }
        }
        if is_b_slice {
            let mod_flag_l1 = r.read_bit()? == 1;
            if mod_flag_l1 {
                loop {
                    let op = r.read_ue()?;
                    if op == 3 {
                        break;
                    }
                    let _val = r.read_ue()?;
                    if r.bits_remaining() < 4 {
                        break;
                    }
                }
            }
        }
    }

    // pred_weight_table() — comes BEFORE dec_ref_pic_marking per H.264 spec 7.3.3
    let weight_table =
        if (is_p_slice && pps.weighted_pred_flag) || (is_b_slice && pps.weighted_bipred_idc == 1) {
            Some(parse_weight_table(
                r,
                slice_type,
                pps.num_ref_idx_l0_default_active,
                pps.num_ref_idx_l1_default_active,
                sps.chroma_format_idc,
            )?)
        } else {
            None
        };

    // dec_ref_pic_marking()
    if is_idr {
        let _no_output_of_prior_pics = r.read_bit()?;
        let _long_term_reference_flag = r.read_bit()?;
    } else {
        let adaptive = r.read_bit()? == 1;
        if adaptive {
            loop {
                let op = r.read_ue()?;
                if op == 0 {
                    break;
                }
                match op {
                    1 => {
                        let _ = r.read_ue()?;
                    }
                    2 => {
                        let _ = r.read_ue()?;
                    }
                    3 => {
                        let _ = r.read_ue()?;
                        let _ = r.read_ue()?;
                    }
                    4 => {
                        let _ = r.read_ue()?;
                    }
                    5 => {}
                    6 => {
                        let _ = r.read_ue()?;
                    }
                    _ => break,
                }
                if r.bits_remaining() < 4 {
                    break;
                }
            }
        }
    }

    // cabac_init_idc (for CABAC slices, non-I only)
    if pps.entropy_coding_mode_flag && !is_i_slice {
        let _cabac_init_idc = r.read_ue()?;
    }

    let slice_qp_delta = r.read_se()?;
    let qp = pps.pic_init_qp + slice_qp_delta;

    // deblocking filter parameters (when pps flag is set)
    if pps.deblocking_filter_control_present_flag {
        let disable_deblocking = r.read_ue()?;
        if disable_deblocking != 1 {
            let _alpha_offset = r.read_se()?;
            let _beta_offset = r.read_se()?;
        }
    }

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type,
        pps_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        qp,
        weight_table,
    })
}
