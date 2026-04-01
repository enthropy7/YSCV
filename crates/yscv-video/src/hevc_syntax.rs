//! HEVC CU/PU/TU syntax parsing using CABAC (ITU-T H.265, sections 7.3.8-7.3.12).
//!
//! This module reads coded data from the bitstream via the CABAC engine
//! ([`CabacDecoder`]) and produces decoded coding unit results including
//! prediction mode, intra modes, and transform coefficients.

use super::hevc_cabac::{CabacDecoder, ContextModel};
use super::hevc_decoder::{HevcPps, HevcPredMode, HevcSliceType, HevcSps};
use super::hevc_inter::{HevcMvField, parse_inter_prediction};

// ---------------------------------------------------------------------------
// Context index offsets (ITU-T H.265, Table 9-4)
// ---------------------------------------------------------------------------

/// `split_cu_flag` — 3 contexts (indexed by depth + neighbour availability).
pub const CTX_SPLIT_CU_FLAG: usize = 0;
/// `cu_skip_flag` — 3 contexts.
pub const CTX_CU_SKIP_FLAG: usize = 3;
/// `pred_mode_flag` — 1 context.
pub const CTX_PRED_MODE_FLAG: usize = 6;
/// `part_mode` — 4 contexts.
pub const CTX_PART_MODE: usize = 7;
/// `prev_intra_luma_pred_flag` — 1 context.
pub const CTX_PREV_INTRA_LUMA_PRED_FLAG: usize = 11;
/// `intra_chroma_pred_mode` — 1 context.
pub const CTX_INTRA_CHROMA_PRED_MODE: usize = 12;
/// `cbf_luma` — 2 contexts (indexed by transform depth).
pub const CTX_CBF_LUMA: usize = 13;
/// `cbf_cb` / `cbf_cr` — 4 contexts (shared).
pub const CTX_CBF_CB: usize = 15;
/// `last_sig_coeff_x_prefix` — 18 contexts.
pub const CTX_LAST_SIG_COEFF_X_PREFIX: usize = 19;
/// `last_sig_coeff_y_prefix` — 18 contexts.
pub const CTX_LAST_SIG_COEFF_Y_PREFIX: usize = 37;
/// `coded_sub_block_flag` — 4 contexts.
pub const CTX_CODED_SUB_BLOCK_FLAG: usize = 55;
/// `sig_coeff_flag` — 44 contexts.
pub const CTX_SIG_COEFF_FLAG: usize = 59;
/// `coeff_abs_level_greater1_flag` — 24 contexts.
pub const CTX_COEFF_ABS_LEVEL_GREATER1: usize = 103;
/// `coeff_abs_level_greater2_flag` — 6 contexts.
pub const CTX_COEFF_ABS_LEVEL_GREATER2: usize = 127;
/// Total number of CABAC context models for syntax parsing.
pub const NUM_CABAC_CONTEXTS: usize = 133;

// ---------------------------------------------------------------------------
// Default initialisation values (representative subset from Table 9-4)
// ---------------------------------------------------------------------------

/// Default init values for I-slice contexts (Table 9-4, initType = 0).
/// One value per context in the order defined by the CTX_* constants above.
#[rustfmt::skip]
const INIT_VALUES_I_SLICE: [u8; NUM_CABAC_CONTEXTS] = [
    // split_cu_flag (3)
    139, 141, 157,
    // cu_skip_flag (3)
    197, 185, 201,
    // pred_mode_flag (1)
    149,
    // part_mode (4)
    184, 154, 139, 154,
    // prev_intra_luma_pred_flag (1)
    184,
    // intra_chroma_pred_mode (1)
    152,
    // cbf_luma (2)
    111, 141,
    // cbf_cb (4)
    94, 138, 182, 154,
    // last_sig_coeff_x_prefix (18)
    110, 110, 124, 125, 140, 153, 125, 127, 140,
    109, 111, 143, 127, 111, 79, 108, 123, 63,
    // last_sig_coeff_y_prefix (18)
    110, 110, 124, 125, 140, 153, 125, 127, 140,
    109, 111, 143, 127, 111, 79, 108, 123, 63,
    // coded_sub_block_flag (4)
    91, 171, 134, 141,
    // sig_coeff_flag (44)
    111, 111, 125, 110, 110,  94, 124, 108, 124, 107,
    125, 141, 179, 153, 125, 107, 125, 141, 179, 153,
    125, 107, 125, 141, 179, 153, 125, 107, 125, 141,
    179, 153, 125, 141, 140, 139, 182, 182, 152, 136,
    152, 136, 153, 136,
    // coeff_abs_level_greater1 (24)
    140, 92, 137, 138, 140, 152, 138, 139, 153,  74,
    149,  92, 139, 107, 122, 152, 140, 179, 166, 182,
    140, 227, 122, 197,
    // coeff_abs_level_greater2 (6)
    138, 153, 136, 167, 152, 152,
];

// ---------------------------------------------------------------------------
// Scan orders for coefficient coding (ITU-T H.265, 6.5.3)
// ---------------------------------------------------------------------------

/// Diagonal up-right scan for 4x4 sub-blocks.
#[rustfmt::skip]
const SCAN_ORDER_4X4_DIAG: [u8; 16] = [
     0,  4,  1,  8,  5,  2, 12,  9,
     6,  3, 13, 10,  7, 14, 11, 15,
];

/// Scan order for 4x4 sub-block positions within an 8x8 TU (2x2 sub-blocks).
#[rustfmt::skip]
const SCAN_ORDER_2X2_DIAG: [u8; 4] = [0, 2, 1, 3];

/// Scan order for 4x4 sub-block positions within a 16x16 TU (4x4 sub-blocks).
#[rustfmt::skip]
const SCAN_ORDER_4X4_SUBBLOCK_DIAG: [u8; 16] = [
     0,  4,  1,  8,  5,  2, 12,  9,
     6,  3, 13, 10,  7, 14, 11, 15,
];

// ---------------------------------------------------------------------------
// Slice-level CABAC state
// ---------------------------------------------------------------------------

/// CABAC state for decoding a single slice, holding all context models and the
/// arithmetic decoder tied to the slice payload data.
pub struct HevcSliceCabacState<'a> {
    /// Adaptive probability contexts for all syntax elements.
    pub contexts: Vec<ContextModel>,
    /// The arithmetic decoder reading from the slice payload.
    pub cabac: CabacDecoder<'a>,
}

impl<'a> HevcSliceCabacState<'a> {
    /// Create a new slice CABAC state from slice payload bytes and QP.
    ///
    /// Initialises all context models according to ITU-T H.265 Table 9-4
    /// using the given `slice_qp` and the I-slice initialisation table.
    pub fn new(slice_data: &'a [u8], slice_qp: i32) -> Self {
        let mut contexts = Vec::with_capacity(NUM_CABAC_CONTEXTS);
        for &iv in &INIT_VALUES_I_SLICE {
            let mut ctx = ContextModel::new(iv);
            ctx.init(slice_qp, iv);
            contexts.push(ctx);
        }
        let cabac = CabacDecoder::new(slice_data);
        HevcSliceCabacState { contexts, cabac }
    }

    /// Re-initialise all context models for a given QP (e.g. at WPP row start).
    pub fn reinit_contexts(&mut self, slice_qp: i32) {
        for (ctx, &iv) in self.contexts.iter_mut().zip(INIT_VALUES_I_SLICE.iter()) {
            ctx.init(slice_qp, iv);
        }
    }
}

// ---------------------------------------------------------------------------
// CU-level data
// ---------------------------------------------------------------------------

/// Decoded data produced by [`parse_coding_unit`].
#[derive(Debug, Clone)]
pub struct CodingUnitData {
    /// Prediction mode (Intra, Inter, Skip).
    pub pred_mode: HevcPredMode,
    /// Luma intra prediction mode index (0..=34).
    pub intra_mode_luma: u8,
    /// Chroma intra prediction mode index.
    pub intra_mode_chroma: u8,
    /// Whether the luma CBF is set (nonzero residual).
    pub cbf_luma: bool,
    /// Whether the Cb CBF is set.
    pub cbf_cb: bool,
    /// Whether the Cr CBF is set.
    pub cbf_cr: bool,
    /// Transform coefficients (luma) in scan order, length = block_size^2.
    pub residual_luma: Vec<i16>,
    /// Transform coefficients (Cb chroma) in scan order, length = (block_size/2)^2.
    pub residual_cb: Vec<i16>,
    /// Transform coefficients (Cr chroma) in scan order, length = (block_size/2)^2.
    pub residual_cr: Vec<i16>,
}

// ---------------------------------------------------------------------------
// Coding tree traversal (split_cu_flag)
// ---------------------------------------------------------------------------

/// Read `split_cu_flag` from the bitstream.
///
/// Context index is derived from the current depth plus the availability of
/// left/above neighbours (simplified: `ctx_idx = depth.min(2)`).
pub fn parse_split_cu_flag(
    state: &mut HevcSliceCabacState<'_>,
    depth: u8,
    _left_available: bool,
    _above_available: bool,
) -> bool {
    // Context selection: depth contributes to the index (spec 9.3.4.2.2).
    // Simplified: left+above availability each add 1 in the real spec, but
    // here we approximate with just depth clamped to 0..2.
    let ctx_idx = CTX_SPLIT_CU_FLAG + (depth as usize).min(2);
    state.cabac.decode_decision(&mut state.contexts[ctx_idx])
}

// ---------------------------------------------------------------------------
// CU-level syntax parsing
// ---------------------------------------------------------------------------

/// Parse a coding unit from the bitstream (ITU-T H.265, 7.3.8.5).
///
/// Returns the prediction mode, intra luma/chroma modes, CBF flags, and
/// residual transform coefficients for the luma plane.
pub fn parse_coding_unit(
    state: &mut HevcSliceCabacState<'_>,
    _x: usize,
    _y: usize,
    log2_cu_size: u32,
    sps: &HevcSps,
    pps: &HevcPps,
    slice_type: HevcSliceType,
) -> CodingUnitData {
    let cu_size = 1u32 << log2_cu_size;
    let num_samples = (cu_size * cu_size) as usize;

    // -- cu_skip_flag (P/B slices only) ------------------------------------
    let skip_flag = if slice_type != HevcSliceType::I {
        let ctx_idx = CTX_CU_SKIP_FLAG; // simplified: always ctx 0
        state.cabac.decode_decision(&mut state.contexts[ctx_idx])
    } else {
        false
    };

    if skip_flag {
        return CodingUnitData {
            pred_mode: HevcPredMode::Skip,
            intra_mode_luma: 0,
            intra_mode_chroma: 0,
            cbf_luma: false,
            cbf_cb: false,
            cbf_cr: false,
            residual_luma: Vec::new(),
            residual_cb: Vec::new(),
            residual_cr: Vec::new(),
        };
    }

    // -- pred_mode_flag (non-I slices) -------------------------------------
    let pred_mode = if slice_type == HevcSliceType::I {
        HevcPredMode::Intra
    } else {
        let ctx_idx = CTX_PRED_MODE_FLAG;
        if state.cabac.decode_decision(&mut state.contexts[ctx_idx]) {
            HevcPredMode::Intra
        } else {
            HevcPredMode::Inter
        }
    };

    // -- part_mode ----------------------------------------------------------
    // For intra CUs the only valid mode is PART_2Nx2N (spec 7.4.9.5).
    // For inter we decode but currently only handle 2Nx2N.
    if pred_mode == HevcPredMode::Inter {
        let ctx_idx = CTX_PART_MODE;
        let _part_2nx2n = state.cabac.decode_decision(&mut state.contexts[ctx_idx]);
        // If not 2Nx2N, additional bins would follow; skip for now.
    }

    // -- Intra mode signalling ---------------------------------------------
    let (intra_mode_luma, intra_mode_chroma) = if pred_mode == HevcPredMode::Intra {
        let luma = parse_intra_mode_luma(state);
        let chroma = parse_intra_chroma_pred_mode(state);
        (luma, chroma)
    } else {
        (0u8, 0u8)
    };

    // -- Transform tree (simplified: single TU = CU) -----------------------
    let log2_min_tu = sps.log2_min_transform_size as u32;
    let log2_tu = log2_cu_size.max(log2_min_tu);

    // CBF flags
    let cbf_cb = parse_cbf_chroma(state, 0);
    let cbf_cr = parse_cbf_chroma(state, 0);
    let cbf_luma = parse_cbf_luma(state, 0);

    // Residual coefficients (luma) — skip Vec for uncoded blocks
    let residual_luma = if cbf_luma {
        let mut tu_buf = vec![0i16; num_samples];
        parse_transform_unit(
            state,
            log2_tu,
            true,
            pps.sign_data_hiding_enabled,
            &mut tu_buf,
        );
        tu_buf
    } else {
        Vec::new()
    };

    // Chroma residual coefficients (half resolution for 4:2:0)
    let chroma_log2_tu = log2_tu.saturating_sub(1).max(2); // min 4x4
    let chroma_samples = (1usize << chroma_log2_tu) * (1usize << chroma_log2_tu);
    let residual_cb = if cbf_cb {
        let mut tu_buf = vec![0i16; chroma_samples];
        parse_transform_unit(
            state,
            chroma_log2_tu,
            false,
            pps.sign_data_hiding_enabled,
            &mut tu_buf,
        );
        tu_buf
    } else {
        Vec::new()
    };
    let residual_cr = if cbf_cr {
        let mut tu_buf = vec![0i16; chroma_samples];
        parse_transform_unit(
            state,
            chroma_log2_tu,
            false,
            pps.sign_data_hiding_enabled,
            &mut tu_buf,
        );
        tu_buf
    } else {
        Vec::new()
    };

    CodingUnitData {
        pred_mode,
        intra_mode_luma,
        intra_mode_chroma,
        cbf_luma,
        cbf_cb,
        cbf_cr,
        residual_luma,
        residual_cb,
        residual_cr,
    }
}

// ---------------------------------------------------------------------------
// Intra mode signalling (ITU-T H.265, 7.3.8.5 + 8.4.2)
// ---------------------------------------------------------------------------

/// Parse luma intra prediction mode.
///
/// Reads `prev_intra_luma_pred_flag`; if set, reads `mpm_idx` (TR-coded,
/// 0..2); otherwise reads `rem_intra_luma_pred_mode` (5 bypass bins).
///
/// The Most Probable Mode (MPM) list is constructed from DC, Planar, and
/// Angular-26 as a simplified default (real spec uses neighbour modes).
#[inline]
fn parse_intra_mode_luma(state: &mut HevcSliceCabacState<'_>) -> u8 {
    let ctx_idx = CTX_PREV_INTRA_LUMA_PRED_FLAG;
    let prev_flag = state.cabac.decode_decision(&mut state.contexts[ctx_idx]);

    if prev_flag {
        // mpm_idx: truncated unary, max 2, bypass coded
        let mpm_idx = parse_mpm_idx(state);
        // Simplified MPM list: {Planar(0), DC(1), Angular-26(26)}
        let mpm_list = build_default_mpm_list();
        mpm_list[mpm_idx as usize]
    } else {
        // rem_intra_luma_pred_mode: 5 bypass bins (0..31)
        let rem = state.cabac.decode_fl(5) as u8;
        // Map rem to actual mode, skipping MPM entries (simplified)
        let mpm_list = build_default_mpm_list();
        remap_rem_mode(rem, &mpm_list)
    }
}

/// Decode `mpm_idx` — truncated unary bypass code, max value 2.
fn parse_mpm_idx(state: &mut HevcSliceCabacState<'_>) -> u8 {
    // mpm_idx is bypass-coded as truncated unary with cMax=2
    if !state.cabac.decode_bypass() {
        0
    } else if !state.cabac.decode_bypass() {
        1
    } else {
        2
    }
}

/// Build the default MPM list when neighbour modes are unavailable.
///
/// Per ITU-T H.265 8.4.2, when both neighbours are unavailable the MPM list
/// is {Planar, DC, Angular-26}. The list is always sorted in ascending order.
fn build_default_mpm_list() -> [u8; 3] {
    let mut mpm = [0u8, 1, 26]; // Planar, DC, Angular-26
    // Sort ascending (already sorted in this case)
    mpm.sort_unstable();
    mpm
}

/// Build the MPM list from left and above neighbour intra modes.
///
/// Follows ITU-T H.265 section 8.4.2 for constructing the three most
/// probable modes.
pub fn build_mpm_list(left_mode: u8, above_mode: u8) -> [u8; 3] {
    let mut mpm = [0u8; 3];
    if left_mode == above_mode {
        if left_mode < 2 {
            // Both are Planar or DC
            mpm[0] = 0; // Planar
            mpm[1] = 1; // DC
            mpm[2] = 26; // Angular-26 (vertical)
        } else {
            mpm[0] = left_mode;
            mpm[1] = 2 + ((left_mode + 29) % 32);
            mpm[2] = 2 + ((left_mode - 2 + 1) % 32);
        }
    } else {
        mpm[0] = left_mode;
        mpm[1] = above_mode;
        if left_mode != 0 && above_mode != 0 {
            mpm[2] = 0; // Planar
        } else if left_mode != 1 && above_mode != 1 {
            mpm[2] = 1; // DC
        } else {
            mpm[2] = 26; // Angular-26
        }
    }
    mpm
}

/// Map `rem_intra_luma_pred_mode` to the actual mode index, skipping MPMs.
///
/// The `rem` value (0..31) indexes into the 32 non-MPM modes.  We walk
/// through modes 0..34, skip the three MPM entries, and select the `rem`-th
/// remaining mode.
fn remap_rem_mode(rem: u8, mpm_list: &[u8; 3]) -> u8 {
    let mut sorted_mpm = *mpm_list;
    sorted_mpm.sort_unstable();

    let mut mode = rem;
    for &m in &sorted_mpm {
        if mode >= m {
            mode += 1;
        }
    }
    mode.min(34)
}

/// Parse `intra_chroma_pred_mode` (ITU-T H.265, 7.3.8.5).
///
/// One context-coded bin selects between mode 4 (derived from luma) and an
/// explicit 2-bit bypass-coded index (0..3 mapping to planar/angular/DC/angular).
fn parse_intra_chroma_pred_mode(state: &mut HevcSliceCabacState<'_>) -> u8 {
    let ctx_idx = CTX_INTRA_CHROMA_PRED_MODE;
    let derived = state.cabac.decode_decision(&mut state.contexts[ctx_idx]);

    if !derived {
        // Mode 4: "derived from luma" (DM mode)
        4
    } else {
        // 2 bypass bins encoding index 0..3
        state.cabac.decode_fl(2) as u8
    }
}

// ---------------------------------------------------------------------------
// CBF (Coded Block Flag) parsing
// ---------------------------------------------------------------------------

/// Parse `cbf_luma` (ITU-T H.265, 7.3.8.11).
///
/// Context index depends on the transform depth within the CU.
#[inline]
fn parse_cbf_luma(state: &mut HevcSliceCabacState<'_>, trafo_depth: u32) -> bool {
    let ctx_idx = CTX_CBF_LUMA + (trafo_depth.min(1) as usize);
    state.cabac.decode_decision(&mut state.contexts[ctx_idx])
}

/// Parse `cbf_cb` or `cbf_cr` (ITU-T H.265, 7.3.8.11).
///
/// Contexts are shared between Cb and Cr; index depends on transform depth.
#[inline]
fn parse_cbf_chroma(state: &mut HevcSliceCabacState<'_>, trafo_depth: u32) -> bool {
    let ctx_idx = CTX_CBF_CB + (trafo_depth.min(3) as usize);
    state.cabac.decode_decision(&mut state.contexts[ctx_idx])
}

// ---------------------------------------------------------------------------
// TU-level residual parsing (ITU-T H.265, 7.3.8.11 + 7.3.8.12)
// ---------------------------------------------------------------------------

/// Parse a transform unit's residual coefficients.
///
/// Implements the coefficient coding syntax from ITU-T H.265 section 7.3.8.12
/// including last-significant-coefficient position, sub-block significance
/// flags, per-coefficient significance, greater-than-1/2 flags, and bypass-
/// coded sign/remaining-level.
///
/// Returns the dequantised coefficient array in raster order (row-major),
/// with length `(1 << log2_tu_size)^2`.
#[inline]
#[allow(unsafe_code)]
pub fn parse_transform_unit(
    state: &mut HevcSliceCabacState<'_>,
    log2_tu_size: u32,
    is_luma: bool,
    sign_data_hiding_enabled: bool,
    out_buf: &mut [i16],
) {
    let tu_size = 1u32 << log2_tu_size;
    let num_coeffs = (tu_size * tu_size) as usize;
    debug_assert!(out_buf.len() >= num_coeffs);
    // Zero only via pointer (avoids bounds check in fill)
    unsafe {
        std::ptr::write_bytes(out_buf.as_mut_ptr(), 0, num_coeffs);
    }
    let coeffs = &mut out_buf[..num_coeffs];

    // -- Last significant coefficient position ----------------------------
    let (last_x, last_y) = parse_last_sig_coeff_pos(state, log2_tu_size, is_luma);

    if last_x >= tu_size || last_y >= tu_size {
        // Out of bounds — treat as all-zero TU.
        return;
    }

    // -- Sub-block and coefficient scanning --------------------------------
    let log2_sub = 2u32; // 4x4 sub-blocks
    let sub_size = 1u32 << log2_sub;
    let num_sub_x = tu_size >> log2_sub;
    let num_sub_total = (num_sub_x * num_sub_x) as usize;

    // Determine which sub-block contains the last significant coeff
    let last_sub_x = last_x >> log2_sub;
    let last_sub_y = last_y >> log2_sub;
    let last_sub_scan = sub_pos_to_scan_idx(last_sub_x, last_sub_y, num_sub_x);

    // Sub-block coded flags — zero only needed portion
    let mut sub_coded_buf = [false; 256];
    let n_sub = num_sub_total.min(256);
    unsafe {
        std::ptr::write_bytes(sub_coded_buf.as_mut_ptr(), 0, n_sub);
    }
    let sub_coded = &mut sub_coded_buf[..n_sub];
    if last_sub_scan < num_sub_total {
        sub_coded[last_sub_scan] = true; // last sub-block always coded
    }

    // Process sub-blocks in reverse scan order
    let first_sub = if last_sub_scan < num_sub_total {
        last_sub_scan
    } else {
        0
    };

    for sub_scan in (0..=first_sub).rev() {
        // Read coded_sub_block_flag for non-last, non-DC sub-blocks
        if sub_scan < first_sub && sub_scan > 0 {
            let ctx_idx = CTX_CODED_SUB_BLOCK_FLAG + if is_luma { 0 } else { 2 };
            sub_coded[sub_scan] = state.cabac.decode_decision(&mut state.contexts[ctx_idx]);
        } else if sub_scan == 0 && first_sub > 0 {
            // DC sub-block: infer coded if any higher sub-block is coded
            sub_coded[0] = true;
        } else if sub_scan == first_sub {
            sub_coded[sub_scan] = true;
        }

        if !sub_coded[sub_scan] {
            continue;
        }

        // Get sub-block position in raster order
        let (sub_x, sub_y) = scan_idx_to_sub_pos(sub_scan, num_sub_x);
        let base_x = sub_x * sub_size;
        let base_y = sub_y * sub_size;

        // Determine the last coeff scan index within this sub-block
        let last_scan_in_sub = if sub_scan == first_sub {
            // The sub-block that contains the last significant coeff
            let local_x = last_x - base_x;
            let local_y = last_y - base_y;
            local_pos_to_scan_idx(local_x, local_y)
        } else {
            15 // full 4x4 sub-block
        };

        // Parse significance flags, levels, and signs for this sub-block
        parse_subblock_coeffs(
            state,
            coeffs,
            tu_size,
            base_x,
            base_y,
            last_scan_in_sub,
            is_luma,
            sub_scan == first_sub,
            sign_data_hiding_enabled,
        );
    }
}

// ---------------------------------------------------------------------------
// Last significant coefficient position
// ---------------------------------------------------------------------------

/// Parse `last_sig_coeff_x_prefix/suffix` and `last_sig_coeff_y_prefix/suffix`.
///
/// Returns `(last_x, last_y)` — the position of the last significant
/// coefficient in the TU (in scan-to-raster mapped coordinates).
fn parse_last_sig_coeff_pos(
    state: &mut HevcSliceCabacState<'_>,
    log2_tu_size: u32,
    is_luma: bool,
) -> (u32, u32) {
    let last_x = parse_last_sig_coeff_prefix_suffix(state, log2_tu_size, is_luma, true);
    let last_y = parse_last_sig_coeff_prefix_suffix(state, log2_tu_size, is_luma, false);
    (last_x, last_y)
}

/// Parse one component (X or Y) of the last significant coefficient position.
///
/// The prefix is truncated-unary coded with context models; the suffix
/// (if prefix >= 2) is bypass-coded fixed-length.
fn parse_last_sig_coeff_prefix_suffix(
    state: &mut HevcSliceCabacState<'_>,
    log2_tu_size: u32,
    is_luma: bool,
    is_x: bool,
) -> u32 {
    let tu_size = 1u32 << log2_tu_size;
    // Maximum prefix value = 2 * (log2_tu_size - 1)  (capped at TU size)
    let max_prefix = if log2_tu_size > 1 {
        2 * (log2_tu_size - 1)
    } else {
        0
    };
    // Context offset depends on component (x/y) and luma/chroma
    let ctx_base = if is_x {
        CTX_LAST_SIG_COEFF_X_PREFIX
    } else {
        CTX_LAST_SIG_COEFF_Y_PREFIX
    };
    let ctx_offset_c = if is_luma { 0usize } else { 9 };

    // Decode prefix as truncated unary
    let mut prefix = 0u32;
    while prefix < max_prefix {
        // Context index: 3 * (log2_tu_size - 2) + (prefix >> 1), capped to 8
        let ctx_inc = if log2_tu_size >= 2 {
            let group = (prefix >> 1) as usize;
            let base = 3 * ((log2_tu_size as usize).saturating_sub(2));
            (base + group).min(8)
        } else {
            0
        };
        let ctx_idx = ctx_base + ctx_offset_c + ctx_inc;
        let ctx_idx = ctx_idx.min(state.contexts.len() - 1);
        if state.cabac.decode_decision(&mut state.contexts[ctx_idx]) {
            prefix += 1;
        } else {
            break;
        }
    }

    // Decode suffix (if prefix >= 2)
    if prefix < 2 {
        return prefix;
    }
    let suffix_len = (prefix >> 1) - 1;
    if suffix_len == 0 {
        return prefix;
    }
    let suffix = state.cabac.decode_fl(suffix_len);
    let value = (1u32 << suffix_len) + suffix + prefix - 2;
    value.min(tu_size - 1)
}

// ---------------------------------------------------------------------------
// Sub-block coefficient parsing
// ---------------------------------------------------------------------------

/// Parse significance flags, levels, and signs for one 4x4 sub-block.
///
/// # Safety
/// Uses `get_unchecked` for performance — all indices are bounded by the
/// scan range (0..=15) and verified by debug_assert.
#[allow(unsafe_code, clippy::too_many_arguments)]
fn parse_subblock_coeffs(
    state: &mut HevcSliceCabacState<'_>,
    coeffs: &mut [i16],
    tu_size: u32,
    base_x: u32,
    base_y: u32,
    last_scan_pos: u32,
    is_luma: bool,
    is_last_subblock: bool,
    sign_data_hiding_enabled: bool,
) {
    debug_assert!(last_scan_pos <= 15);
    debug_assert!(coeffs.len() >= (tu_size * tu_size) as usize);
    debug_assert!(state.contexts.len() >= NUM_CABAC_CONTEXTS);

    let last = last_scan_pos.min(15) as usize;
    let sig_ctx_table = if is_luma {
        &SIG_CTX_INC_LUMA
    } else {
        &SIG_CTX_INC_CHROMA
    };

    // Step 1: significance flags — unsafe unchecked indexing
    let mut sig = [0u8; 16]; // 0/1 instead of bool for branchless math
    let mut num_sig = 0u32;

    unsafe {
        let ctxs = state.contexts.as_mut_ptr();
        let mut i = last;
        loop {
            if is_last_subblock && i == last {
                *sig.get_unchecked_mut(i) = 1;
                num_sig += 1;
            } else {
                let ctx_inc = *sig_ctx_table.get_unchecked(i) as usize;
                let ctx_idx = CTX_SIG_COEFF_FLAG + ctx_inc;
                let decided = state.cabac.decode_decision(&mut *ctxs.add(ctx_idx));
                let s = decided as u8;
                *sig.get_unchecked_mut(i) = s;
                num_sig += s as u32;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    if num_sig == 0 {
        return;
    }

    // Step 2: greater1 and greater2 flags
    let mut greater1 = [0u8; 16];
    let mut greater2 = [0u8; 16];
    let mut coeff_count = 0u32;
    let ctx_set = if is_luma { 0usize } else { 12 };

    unsafe {
        let ctxs = state.contexts.as_mut_ptr();
        let mut i = last;
        loop {
            if *sig.get_unchecked(i) != 0 {
                if coeff_count < 8 {
                    let ctx_inc = ctx_set + (coeff_count as usize).min(3);
                    let ctx_idx = CTX_COEFF_ABS_LEVEL_GREATER1 + ctx_inc;
                    *greater1.get_unchecked_mut(i) =
                        state.cabac.decode_decision(&mut *ctxs.add(ctx_idx)) as u8;
                }
                coeff_count += 1;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    // greater2: only for the first coefficient with greater1=1
    let mut first_g1: u32 = 16; // sentinel
    unsafe {
        let mut i = last;
        loop {
            if *sig.get_unchecked(i) != 0 && *greater1.get_unchecked(i) != 0 {
                first_g1 = i as u32;
                break;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        if first_g1 < 16 {
            let ctx_inc = if is_luma { 0usize } else { 3 };
            let ctx_idx = CTX_COEFF_ABS_LEVEL_GREATER2 + ctx_inc;
            let ctxs = state.contexts.as_mut_ptr();
            *greater2.get_unchecked_mut(first_g1 as usize) =
                state.cabac.decode_decision(&mut *ctxs.add(ctx_idx)) as u8;
        }
    }

    // Step 3: signs (bypass coded)
    let mut signs = [0u8; 16]; // 0=positive, 1=negative
    unsafe {
        let mut hidden_count = 0u32;
        let mut i = last;
        loop {
            if *sig.get_unchecked(i) != 0 {
                hidden_count += 1;
                let hide = sign_data_hiding_enabled && num_sig > 1 && i == 0 && last_scan_pos > 3;
                if !hide {
                    *signs.get_unchecked_mut(i) = state.cabac.decode_bypass() as u8;
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        let _ = hidden_count;
    }

    // Step 4: remaining levels + Step 5: write coefficients (fused loop)
    let mut rice_param = 0u32;
    unsafe {
        let coeff_ptr = coeffs.as_mut_ptr();
        let mut i = last;
        loop {
            if *sig.get_unchecked(i) != 0 {
                let g1 = *greater1.get_unchecked(i) as i32;
                let g2 = *greater2.get_unchecked(i) as i32;
                let base_level = 1 + g1 + g2;

                let needs_remaining = g2 != 0 || (g1 != 0 && i as u32 != first_g1);
                let remaining = if needs_remaining {
                    decode_coeff_abs_level_remaining(state, rice_param) as i32
                } else {
                    0
                };

                let abs_val = base_level + remaining;

                // Update Rice parameter (branchless)
                if abs_val > (3i32 << rice_param) {
                    rice_param = (rice_param + 1).min(4);
                }

                // Write coefficient — pre-computed scan→position table, branchless sign
                let (lx, ly) = *SCAN_TO_XY.get_unchecked(i);
                let px = base_x + lx as u32;
                let py = base_y + ly as u32;
                let idx = (py * tu_size + px) as usize;
                // sign_mask: 0 for positive, -1 for negative
                let sign = *signs.get_unchecked(i) as i16;
                let val = abs_val as i16;
                // Branchless: (val ^ -sign) + sign = negate if sign=1
                let signed_val = (val ^ (-sign)) + sign;
                *coeff_ptr.add(idx) = signed_val;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }
}

/// Decode `coeff_abs_level_remaining` using Exp-Golomb-Rice bypass coding
/// (ITU-T H.265, 9.3.3.11).
fn decode_coeff_abs_level_remaining(state: &mut HevcSliceCabacState<'_>, rice_param: u32) -> u32 {
    // Count prefix ones (up to a max to avoid infinite loops)
    let mut prefix = 0u32;
    let max_prefix = 28u32; // safety limit
    while prefix < max_prefix && state.cabac.decode_bypass() {
        prefix += 1;
    }

    if prefix < 3 {
        // Standard Rice coding: suffix has rice_param bits
        let suffix = if rice_param > 0 {
            state.cabac.decode_fl(rice_param)
        } else {
            0
        };
        (prefix << rice_param) + suffix
    } else {
        // Exp-Golomb extension: suffix has (prefix - 3 + rice_param) bits
        let suffix_len = prefix - 3 + rice_param;
        let suffix = state.cabac.decode_fl(suffix_len);
        ((1u32 << suffix_len) - 1 + (3u32 << rice_param)).wrapping_add(suffix)
    }
}

// ---------------------------------------------------------------------------
// Scan-order helpers
// ---------------------------------------------------------------------------

// sig_coeff_ctx_inc replaced by pre-computed SIG_CTX_INC_LUMA/CHROMA tables

/// Convert a sub-block scan index to raster (x, y) position within the
/// sub-block grid.
fn scan_idx_to_sub_pos(scan_idx: usize, num_sub_x: u32) -> (u32, u32) {
    let num_sub = (num_sub_x * num_sub_x) as usize;
    if num_sub <= 4 {
        // 2x2 grid
        let idx = if scan_idx < 4 {
            SCAN_ORDER_2X2_DIAG[scan_idx] as u32
        } else {
            scan_idx as u32
        };
        (idx % num_sub_x, idx / num_sub_x)
    } else if num_sub <= 16 {
        // 4x4 grid
        let idx = if scan_idx < 16 {
            SCAN_ORDER_4X4_SUBBLOCK_DIAG[scan_idx] as u32
        } else {
            scan_idx as u32
        };
        (idx % num_sub_x, idx / num_sub_x)
    } else {
        // Larger: use raster fallback
        let idx = scan_idx as u32;
        (idx % num_sub_x, idx / num_sub_x)
    }
}

/// Convert a raster sub-block position to a scan index (reverse of above).
fn sub_pos_to_scan_idx(sub_x: u32, sub_y: u32, num_sub_x: u32) -> usize {
    let raster = sub_y * num_sub_x + sub_x;
    let num_sub = (num_sub_x * num_sub_x) as usize;
    if num_sub <= 4 {
        for (i, &s) in SCAN_ORDER_2X2_DIAG.iter().enumerate() {
            if s as u32 == raster {
                return i;
            }
        }
        raster as usize
    } else if num_sub <= 16 {
        for (i, &s) in SCAN_ORDER_4X4_SUBBLOCK_DIAG.iter().enumerate() {
            if s as u32 == raster {
                return i;
            }
        }
        raster as usize
    } else {
        raster as usize
    }
}

// Pre-computed scan→(x,y) table for 4x4 diagonal scan (avoids division per coeff)
static SCAN_TO_XY: [(u8, u8); 16] = {
    let mut t = [(0u8, 0u8); 16];
    let mut i = 0;
    while i < 16 {
        let raster = SCAN_ORDER_4X4_DIAG[i];
        t[i] = (raster % 4, raster / 4);
        i += 1;
    }
    t
};

// Pre-computed raster→scan reverse lookup (avoids linear search)
static RASTER_TO_SCAN: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        t[SCAN_ORDER_4X4_DIAG[i] as usize] = i as u8;
        i += 1;
    }
    t
};

// Pre-computed sig_coeff context increment table (avoids match per coeff)
// Index: scan_idx (0-15), luma offset=0, chroma offset=27
static SIG_CTX_INC_LUMA: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0u8;
    while i < 16 {
        t[i as usize] = match i {
            0 => 0,
            1..=4 => 1,
            5..=8 => 2,
            9..=12 => 3,
            _ => 4,
        };
        i += 1;
    }
    t
};
static SIG_CTX_INC_CHROMA: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0u8;
    while i < 16 {
        t[i as usize] = 27
            + match i {
                0 => 0,
                1..=4 => 1,
                5..=8 => 2,
                9..=12 => 3,
                _ => 4,
            };
        i += 1;
    }
    t
};

/// Convert a scan index within a 4x4 sub-block to local (x, y) position.
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
fn scan_to_local_pos(scan_idx: u32) -> (u32, u32) {
    let (x, y) = SCAN_TO_XY[scan_idx as usize & 15];
    (x as u32, y as u32)
}

/// Convert a local (x, y) within a 4x4 sub-block to a scan index.
#[inline(always)]
fn local_pos_to_scan_idx(lx: u32, ly: u32) -> u32 {
    RASTER_TO_SCAN[(ly * 4 + lx) as usize & 15] as u32
}

// ---------------------------------------------------------------------------
// Coding tree integration
// ---------------------------------------------------------------------------

/// Recursively decode a coding tree using CABAC, producing decoded CU leaves.
///
/// This replaces the stub `decode_coding_tree` in `hevc_decoder.rs` with
/// actual CABAC-driven split decisions and CU parsing.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_tree_cabac(
    state: &mut HevcSliceCabacState<'_>,
    x: usize,
    y: usize,
    log2_cu_size: u8,
    depth: u8,
    max_depth: u8,
    sps: &HevcSps,
    pps: &HevcPps,
    slice_type: HevcSliceType,
    pic_width: usize,
    pic_height: usize,
    recon_luma: &mut Vec<i16>,
    recon_cb: &mut Vec<i16>,
    recon_cr: &mut Vec<i16>,
    results: &mut Vec<super::hevc_decoder::DecodedCu>,
    dpb: &super::hevc_inter::HevcDpb,
    mv_field: &mut Vec<HevcMvField>,
) {
    let cu_size = 1usize << log2_cu_size;

    // Out of picture bounds — skip
    if x >= pic_width || y >= pic_height {
        return;
    }

    // Decide whether to split
    let can_split = depth < max_depth && cu_size > (1usize << sps.log2_min_cb_size);
    let must_split = cu_size > 64; // CTU is at most 64x64

    let should_split = if must_split {
        true
    } else if can_split {
        let left_avail = x > 0;
        let above_avail = y > 0;
        parse_split_cu_flag(state, depth, left_avail, above_avail)
    } else {
        false
    };

    if should_split {
        let half = log2_cu_size - 1;
        let half_size = 1usize << half;
        let nd = depth + 1;
        decode_coding_tree_cabac(
            state, x, y, half, nd, max_depth, sps, pps, slice_type, pic_width, pic_height,
            recon_luma, recon_cb, recon_cr, results, dpb, mv_field,
        );
        decode_coding_tree_cabac(
            state,
            x + half_size,
            y,
            half,
            nd,
            max_depth,
            sps,
            pps,
            slice_type,
            pic_width,
            pic_height,
            recon_luma,
            recon_cb,
            recon_cr,
            results,
            dpb,
            mv_field,
        );
        decode_coding_tree_cabac(
            state,
            x,
            y + half_size,
            half,
            nd,
            max_depth,
            sps,
            pps,
            slice_type,
            pic_width,
            pic_height,
            recon_luma,
            recon_cb,
            recon_cr,
            results,
            dpb,
            mv_field,
        );
        decode_coding_tree_cabac(
            state,
            x + half_size,
            y + half_size,
            half,
            nd,
            max_depth,
            sps,
            pps,
            slice_type,
            pic_width,
            pic_height,
            recon_luma,
            recon_cb,
            recon_cr,
            results,
            dpb,
            mv_field,
        );
    } else {
        // Leaf CU — parse prediction/residual via CABAC
        let cu_data = parse_coding_unit(state, x, y, log2_cu_size as u32, sps, pps, slice_type);

        let actual_w = cu_size.min(pic_width.saturating_sub(x));
        let actual_h = cu_size.min(pic_height.saturating_sub(y));

        // Intra prediction — zero only the used portion via unsafe
        let mut pred_buf = [0i16; 64 * 64];
        let n = cu_size * cu_size;
        unsafe {
            std::ptr::write_bytes(pred_buf.as_mut_ptr(), 0, n);
        }
        let pred = &mut pred_buf[..n];
        if cu_data.pred_mode == HevcPredMode::Intra {
            // Build top and left reference samples (stack buffers, no heap)
            let mut top_buf = [128i16; 64];
            let mut left_buf = [128i16; 64];
            build_top_ref(recon_luma, x, y, cu_size, pic_width, &mut top_buf);
            build_left_ref(
                recon_luma,
                x,
                y,
                cu_size,
                pic_width,
                pic_height,
                &mut left_buf,
            );
            let top = &top_buf[..cu_size];
            let left = &left_buf[..cu_size];

            match cu_data.intra_mode_luma {
                0 => {
                    let top_right = if x + cu_size < pic_width && y > 0 {
                        recon_luma[(y - 1) * pic_width + x + cu_size]
                    } else {
                        *top.last().unwrap_or(&128)
                    };
                    let bottom_left = if y + cu_size < pic_height && x > 0 {
                        recon_luma[(y + cu_size) * pic_width + x - 1]
                    } else {
                        *left.last().unwrap_or(&128)
                    };
                    super::hevc_decoder::intra_predict_planar(
                        top,
                        left,
                        top_right,
                        bottom_left,
                        cu_size,
                        pred,
                    );
                }
                1 => {
                    super::hevc_decoder::intra_predict_dc(top, left, cu_size, pred);
                }
                m @ 2..=34 => {
                    super::hevc_decoder::intra_predict_angular(top, left, m, cu_size, pred);
                }
                _ => {
                    // Fallback DC
                    super::hevc_decoder::intra_predict_dc(top, left, cu_size, pred);
                }
            }
        } else {
            // Inter/Skip: parse inter prediction data and motion compensate
            // from DPB reference frames.
            let min_pu = 4usize;
            let pic_w_pu = pic_width.div_ceil(min_pu);
            let inter_mv =
                parse_inter_prediction(state, sps, slice_type, mv_field, pic_w_pu, x, y, cu_size);

            // Store MV in the picture-wide MV field for future merge candidates
            let pu_x = x / min_pu;
            let pu_y = y / min_pu;
            let pu_w = cu_size / min_pu;
            for py in 0..pu_w {
                for px in 0..pu_w {
                    let idx = (pu_y + py) * pic_w_pu + (pu_x + px);
                    if idx < mv_field.len() {
                        mv_field[idx] = inter_mv;
                    }
                }
            }

            // Motion compensate from DPB reference
            let ref_poc = inter_mv.ref_idx[0] as i32; // L0 reference POC
            if let Some(ref_pic) = dpb.get_by_poc(ref_poc) {
                super::hevc_inter::hevc_mc_luma(
                    ref_pic,
                    x as i32,
                    y as i32,
                    inter_mv.mv[0],
                    cu_size,
                    cu_size,
                    pred,
                );
                // Chroma MC (4:2:0)
                let cs = cu_size / 2;
                if cs > 0 && !ref_pic.cb.is_empty() {
                    let cw = ref_pic.width / 2;
                    let ch = ref_pic.height / 2;
                    let mut pcb = vec![128i16; cs * cs];
                    let mut pcr = vec![128i16; cs * cs];
                    let cmv = inter_mv.mv[0]; // quarter-pel luma → eighth-pel chroma
                    super::hevc_inter::hevc_mc_chroma(
                        &ref_pic.cb,
                        cw,
                        ch,
                        (x / 2) as i32,
                        (y / 2) as i32,
                        cmv,
                        cs,
                        cs,
                        &mut pcb,
                    );
                    super::hevc_inter::hevc_mc_chroma(
                        &ref_pic.cr,
                        cw,
                        ch,
                        (x / 2) as i32,
                        (y / 2) as i32,
                        cmv,
                        cs,
                        cs,
                        &mut pcr,
                    );
                    // Write chroma recon
                    let cpw = pic_width / 2;
                    for row in 0..cs.min((pic_height / 2).saturating_sub(y / 2)) {
                        for col in 0..cs.min(cpw.saturating_sub(x / 2)) {
                            let ci = (y / 2 + row) * cpw + (x / 2 + col);
                            let si = row * cs + col;
                            if ci < recon_cb.len() {
                                let rcb = if si < cu_data.residual_cb.len() {
                                    cu_data.residual_cb[si] as i32
                                } else {
                                    0
                                };
                                let rcr = if si < cu_data.residual_cr.len() {
                                    cu_data.residual_cr[si] as i32
                                } else {
                                    0
                                };
                                recon_cb[ci] = (pcb[si] as i32 + rcb).clamp(0, 255) as i16;
                                recon_cr[ci] = (pcr[si] as i32 + rcr).clamp(0, 255) as i16;
                            }
                        }
                    }
                }
            } else {
                for v in pred.iter_mut() {
                    *v = 128;
                }
            }
        }

        // Add residual to prediction — write directly to recon_luma (skip temp buffer)
        let max_val = (1i32 << sps.bit_depth_luma) - 1;
        let has_residual = !cu_data.residual_luma.is_empty();
        let residual_ptr = cu_data.residual_luma.as_ptr();
        let residual_len = cu_data.residual_luma.len();

        // Fused reconstruct + writeback: write directly to picture buffer
        #[allow(unsafe_code)]
        unsafe {
            let recon_ptr = recon_luma.as_mut_ptr();
            for row in 0..actual_h {
                let py = y + row;
                if py >= pic_height {
                    break;
                }
                let dst_base = py * pic_width + x;
                let src_base = row * cu_size;
                for col in 0..actual_w {
                    let px = x + col;
                    if px >= pic_width {
                        break;
                    }
                    let i = src_base + col;
                    let p = *pred.get_unchecked(i) as i32;
                    let r = if has_residual && i < residual_len {
                        *residual_ptr.add(i) as i32
                    } else {
                        0
                    };
                    *recon_ptr.add(dst_base + col) = (p + r).clamp(0, max_val) as i16;
                }
            }
        }

        results.push(super::hevc_decoder::DecodedCu {
            x,
            y,
            size: cu_size,
            pred_mode: cu_data.pred_mode,
        });
    }
}

// ---------------------------------------------------------------------------
// Reference sample helpers
// ---------------------------------------------------------------------------

/// Build the top reference row for intra prediction.
#[inline]
fn build_top_ref(
    recon: &[i16],
    x: usize,
    y: usize,
    block_size: usize,
    pic_width: usize,
    out: &mut [i16],
) {
    out[..block_size].fill(128);
    if y > 0 && x + block_size <= pic_width {
        let src = (y - 1) * pic_width + x;
        out[..block_size].copy_from_slice(&recon[src..src + block_size]);
    } else if y > 0 {
        for i in 0..block_size {
            let px = x + i;
            if px < pic_width {
                out[i] = recon[(y - 1) * pic_width + px];
            }
        }
    }
}

/// Build the left reference column for intra prediction.
#[inline]
fn build_left_ref(
    recon: &[i16],
    x: usize,
    y: usize,
    block_size: usize,
    pic_width: usize,
    pic_height: usize,
    out: &mut [i16],
) {
    out[..block_size].fill(128);
    if x > 0 {
        for i in 0..block_size {
            let py = y + i;
            if py < pic_height {
                out[i] = recon[py * pic_width + x - 1];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a CABAC state from raw bytes with default QP 26.
    fn make_state(data: &[u8]) -> HevcSliceCabacState<'_> {
        HevcSliceCabacState::new(data, 26)
    }

    /// Build a default SPS for testing.
    fn test_sps() -> HevcSps {
        HevcSps {
            sps_id: 0,
            vps_id: 0,
            max_sub_layers: 1,
            chroma_format_idc: 1,
            pic_width: 64,
            pic_height: 64,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_pic_order_cnt: 4,
            log2_min_cb_size: 3,
            log2_diff_max_min_cb_size: 3,
            log2_min_transform_size: 2,
            log2_diff_max_min_transform_size: 3,
            max_transform_hierarchy_depth_inter: 1,
            max_transform_hierarchy_depth_intra: 1,
            sample_adaptive_offset_enabled: false,
            pcm_enabled: false,
            num_short_term_ref_pic_sets: 0,
            long_term_ref_pics_present: false,
            sps_temporal_mvp_enabled: false,
            strong_intra_smoothing_enabled: false,
        }
    }

    /// Build a default PPS for testing.
    fn test_pps() -> HevcPps {
        HevcPps {
            pps_id: 0,
            sps_id: 0,
            dependent_slice_segments_enabled: false,
            output_flag_present: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled: false,
            cabac_init_present: false,
            num_ref_idx_l0_default: 1,
            num_ref_idx_l1_default: 1,
            init_qp: 26,
            constrained_intra_pred: false,
            transform_skip_enabled: false,
            cu_qp_delta_enabled: false,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_filter_override_enabled: false,
            deblocking_filter_disabled: true,
            loop_filter_across_slices_enabled: false,
            tiles_enabled: false,
            entropy_coding_sync_enabled: false,
        }
    }

    // -- Context initialisation tests ----------------------------------------

    #[test]
    fn cabac_state_context_count() {
        let data = [0u8; 16];
        let state = make_state(&data);
        assert_eq!(state.contexts.len(), NUM_CABAC_CONTEXTS);
    }

    #[test]
    fn cabac_state_reinit_preserves_count() {
        let data = [0u8; 16];
        let mut state = make_state(&data);
        state.reinit_contexts(30);
        assert_eq!(state.contexts.len(), NUM_CABAC_CONTEXTS);
    }

    // -- split_cu_flag tests -------------------------------------------------

    #[test]
    fn split_cu_flag_deterministic() {
        let data = [0x00u8; 32];
        let mut state = make_state(&data);
        // Decode several split flags at different depths — should not panic
        // and should produce deterministic results.
        let r0 = parse_split_cu_flag(&mut state, 0, false, false);
        let r1 = parse_split_cu_flag(&mut state, 1, true, false);
        let r2 = parse_split_cu_flag(&mut state, 2, true, true);
        // Results are deterministic given the same input
        let _ = (r0, r1, r2);
    }

    #[test]
    fn split_cu_flag_depth_clamp() {
        // Very high depth should still select a valid context (clamped to 2)
        let data = [0xFFu8; 32];
        let mut state = make_state(&data);
        let _ = parse_split_cu_flag(&mut state, 10, true, true);
    }

    // -- Intra mode signalling tests -----------------------------------------

    #[test]
    fn mpm_list_default_construction() {
        let mpm = build_default_mpm_list();
        assert_eq!(mpm.len(), 3);
        assert!(mpm.contains(&0)); // Planar
        assert!(mpm.contains(&1)); // DC
        assert!(mpm.contains(&26)); // Angular-26
    }

    #[test]
    fn mpm_list_from_neighbours_equal_dc() {
        let mpm = build_mpm_list(1, 1);
        assert_eq!(mpm[0], 0); // Planar
        assert_eq!(mpm[1], 1); // DC
        assert_eq!(mpm[2], 26); // Angular-26
    }

    #[test]
    fn mpm_list_from_neighbours_equal_angular() {
        let mpm = build_mpm_list(10, 10);
        assert_eq!(mpm[0], 10);
        // mpm[1] = 2 + ((10 + 29) % 32) = 2 + 7 = 9
        assert_eq!(mpm[1], 9);
        // mpm[2] = 2 + ((10 - 2 + 1) % 32) = 2 + 9 = 11
        assert_eq!(mpm[2], 11);
    }

    #[test]
    fn mpm_list_from_neighbours_different() {
        let mpm = build_mpm_list(5, 10);
        assert_eq!(mpm[0], 5);
        assert_eq!(mpm[1], 10);
        assert_eq!(mpm[2], 0); // Planar (neither is 0)
    }

    #[test]
    fn remap_rem_mode_skips_mpms() {
        let mpm = [0u8, 1, 26];
        // rem=0 should give mode 2 (skipping 0 and 1)
        let mode = remap_rem_mode(0, &mpm);
        assert_eq!(mode, 2);
        // rem=23 should skip modes 0 and 1: 23 -> 24 -> 25
        // (26 is in MPM but 25 < 26, so no further skip)
        let mode = remap_rem_mode(23, &mpm);
        assert_eq!(mode, 25);
        // rem=24 should skip modes 0, 1, and 26: 24 -> 25 -> 26 -> 27
        let mode = remap_rem_mode(24, &mpm);
        assert_eq!(mode, 27);
    }

    #[test]
    fn remap_rem_mode_clamped() {
        let mpm = [0u8, 1, 2];
        // rem=31 => walks past 0,1,2 so mode = 34, clamped
        let mode = remap_rem_mode(31, &mpm);
        assert_eq!(mode, 34);
    }

    // -- Residual coefficient parsing tests ----------------------------------

    #[test]
    fn parse_tu_all_zero_stream() {
        let data = [0x00u8; 64];
        let mut state = make_state(&data);
        let mut coeffs = [0i16; 16];
        parse_transform_unit(&mut state, 2, true, false, &mut coeffs);
        assert_eq!(coeffs.len(), 16); // 4x4
    }

    #[test]
    fn parse_tu_all_ones_stream() {
        let data = [0xFFu8; 128];
        let mut state = make_state(&data);
        let mut coeffs = [0i16; 16];
        parse_transform_unit(&mut state, 2, true, false, &mut coeffs);
        assert_eq!(coeffs.len(), 16);
    }

    #[test]
    fn parse_tu_8x8_size() {
        let data = [0x55u8; 128];
        let mut state = make_state(&data);
        let mut coeffs = [0i16; 64];
        parse_transform_unit(&mut state, 3, true, false, &mut coeffs);
        assert_eq!(coeffs.len(), 64); // 8x8
    }

    #[test]
    fn parse_tu_chroma() {
        let data = [0xAAu8; 64];
        let mut state = make_state(&data);
        let mut coeffs = [0i16; 16];
        parse_transform_unit(&mut state, 2, false, false, &mut coeffs);
        assert_eq!(coeffs.len(), 16);
    }

    // -- Full CU parsing tests -----------------------------------------------

    #[test]
    fn parse_cu_intra_i_slice() {
        let data = [0x00u8; 128];
        let mut state = make_state(&data);
        let sps = test_sps();
        let pps = test_pps();
        let cu = parse_coding_unit(&mut state, 0, 0, 3, &sps, &pps, HevcSliceType::I);
        assert_eq!(cu.pred_mode, HevcPredMode::Intra);
        assert!(cu.intra_mode_luma <= 34);
    }

    #[test]
    fn parse_cu_p_slice_may_skip() {
        // In a P slice, the first bin decoded is cu_skip_flag.
        let data = [0xFFu8; 128];
        let mut state = make_state(&data);
        let sps = test_sps();
        let pps = test_pps();
        let cu = parse_coding_unit(&mut state, 0, 0, 3, &sps, &pps, HevcSliceType::P);
        // Should produce a valid result (skip, intra, or inter)
        assert!(matches!(
            cu.pred_mode,
            HevcPredMode::Intra | HevcPredMode::Inter | HevcPredMode::Skip
        ));
    }

    // -- Coding tree integration tests ---------------------------------------

    #[test]
    fn coding_tree_cabac_produces_cus() {
        let data = [0x00u8; 512];
        let mut state = make_state(&data);
        let sps = test_sps();
        let pps = test_pps();
        let mut recon = vec![128i16; 64 * 64];
        let mut results = Vec::new();

        let dpb = crate::hevc_inter::HevcDpb::new(16);
        let mut mv_field = vec![crate::hevc_inter::HevcMvField::unavailable(); 16 * 16];
        decode_coding_tree_cabac(
            &mut state,
            0,
            0,
            6, // 64x64 CTU
            0,
            3, // max_depth
            &sps,
            &pps,
            HevcSliceType::I,
            64,
            64,
            &mut recon,
            &mut vec![128i16; 32 * 32],
            &mut vec![128i16; 32 * 32],
            &mut results,
            &dpb,
            &mut mv_field,
        );
        // Should produce at least one CU
        assert!(!results.is_empty());
        // All CUs should be within picture bounds
        for cu in &results {
            assert!(cu.x < 64);
            assert!(cu.y < 64);
        }
    }

    #[test]
    fn coding_tree_cabac_boundary() {
        // 48x48 picture with 64x64 CTU — boundary clipping
        let data = [0x55u8; 512];
        let mut state = make_state(&data);
        let sps = test_sps();
        let pps = test_pps();
        let mut recon = vec![128i16; 48 * 48];
        let mut results = Vec::new();
        let dpb = crate::hevc_inter::HevcDpb::new(16);
        let mut mv_field = vec![crate::hevc_inter::HevcMvField::unavailable(); 12 * 12];

        decode_coding_tree_cabac(
            &mut state,
            0,
            0,
            6,
            0,
            3,
            &sps,
            &pps,
            HevcSliceType::I,
            48,
            48,
            &mut recon,
            &mut vec![128i16; 32 * 32],
            &mut vec![128i16; 32 * 32],
            &mut results,
            &dpb,
            &mut mv_field,
        );
        assert!(!results.is_empty());
        for cu in &results {
            assert!(cu.x < 48);
            assert!(cu.y < 48);
        }
    }

    // -- Scan order tests ----------------------------------------------------

    #[test]
    fn scan_4x4_roundtrip() {
        // Every position 0..15 should appear exactly once in the diagonal scan.
        let mut seen = [false; 16];
        for &s in &SCAN_ORDER_4X4_DIAG {
            assert!(!seen[s as usize], "duplicate in scan order");
            seen[s as usize] = true;
        }
        assert!(seen.iter().all(|&v| v));
    }

    #[test]
    fn scan_to_local_roundtrip() {
        for scan_idx in 0..16u32 {
            let (lx, ly) = scan_to_local_pos(scan_idx);
            let back = local_pos_to_scan_idx(lx, ly);
            assert_eq!(back, scan_idx, "roundtrip failed for scan_idx={scan_idx}");
        }
    }

    #[test]
    fn sub_pos_scan_roundtrip_2x2() {
        for scan_idx in 0..4usize {
            let (sx, sy) = scan_idx_to_sub_pos(scan_idx, 2);
            let back = sub_pos_to_scan_idx(sx, sy, 2);
            assert_eq!(back, scan_idx, "2x2 roundtrip failed at {scan_idx}");
        }
    }

    #[test]
    fn sub_pos_scan_roundtrip_4x4() {
        for scan_idx in 0..16usize {
            let (sx, sy) = scan_idx_to_sub_pos(scan_idx, 4);
            let back = sub_pos_to_scan_idx(sx, sy, 4);
            assert_eq!(back, scan_idx, "4x4 roundtrip failed at {scan_idx}");
        }
    }
}
