//! H.264 CABAC (Context-based Adaptive Binary Arithmetic Coding) entropy
//! decoder — the exact clause 9.3 engine and syntax-element layer used by the
//! Main/High-profile decode path.
//!
//! The arithmetic engine implements clause 9.3.3.2 (Table 9-45 state
//! transitions, Table 9-44 range LPS values); contexts initialise from the
//! spec's (m, n) tables (9-12..9-33, generated in `h264_cabac_init`); the
//! element decoders below implement the Table 9-34 binarizations and the
//! clause 9.3.3.1 context-increment choreography. Neighbour-dependent
//! `ctxIdxInc` values are derived by the caller (the decoder owns the
//! neighbour grids) and passed in.

// ---------------------------------------------------------------------------
// State-transition tables (H.264 spec Table 9-45)
// ---------------------------------------------------------------------------

/// State transition after decoding the **Most Probable Symbol** (MPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
static TRANSITION_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 62, 63,
];

/// State transition after decoding the **Least Probable Symbol** (LPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
static TRANSITION_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,
     6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18,
    19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29,
    29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36,
    36, 36, 37, 37, 37, 38, 38, 63,
];

// ---------------------------------------------------------------------------
// Range LPS table (H.264 spec Table 9-44)
// ---------------------------------------------------------------------------

/// `RANGE_TABLE[pStateIdx][qRangeIdx]` — the LPS sub-range for each
/// probability state and quarter-range index.
#[rustfmt::skip]
static RANGE_TABLE: [[u16; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [ 95, 116, 137, 158],
    [ 90, 110, 130, 150],
    [ 85, 104, 123, 142],
    [ 81,  99, 117, 135],
    [ 77,  94, 111, 128],
    [ 73,  89, 105, 122],
    [ 69,  85, 100, 116],
    [ 66,  80,  95, 110],
    [ 62,  76,  90, 104],
    [ 59,  72,  86,  99],
    [ 56,  69,  81,  94],
    [ 53,  65,  77,  89],
    [ 51,  62,  73,  85],
    [ 48,  59,  69,  80],
    [ 46,  56,  66,  76],
    [ 43,  53,  63,  72],
    [ 41,  50,  59,  69],
    [ 39,  48,  56,  65],
    [ 37,  45,  54,  62],
    [ 35,  43,  51,  59],
    [ 33,  41,  48,  56],
    [ 32,  39,  46,  53],
    [ 30,  37,  43,  50],
    [ 29,  35,  41,  48],
    [ 27,  33,  39,  45],
    [ 26,  31,  37,  43],
    [ 24,  30,  35,  41],
    [ 23,  28,  33,  39],
    [ 22,  27,  32,  37],
    [ 21,  26,  30,  35],
    [ 20,  24,  29,  33],
    [ 19,  23,  27,  31],
    [ 18,  22,  26,  30],
    [ 17,  21,  25,  28],
    [ 16,  20,  23,  27],
    [ 15,  19,  22,  25],
    [ 14,  18,  21,  24],
    [ 14,  17,  20,  23],
    [ 13,  16,  19,  22],
    [ 12,  15,  18,  21],
    [ 12,  14,  17,  20],
    [ 11,  14,  16,  19],
    [ 11,  13,  15,  18],
    [ 10,  12,  15,  17],
    [ 10,  12,  14,  16],
    [  9,  11,  13,  15],
    [  9,  11,  12,  14],
    [  8,  10,  12,  14],
    [  8,   9,  11,  13],
    [  7,   9,  11,  12],
    [  7,   9,  10,  12],
    [  7,   8,  10,  11],
    [  6,   8,   9,  11],
    [  6,   7,   9,  10],
    [  6,   7,   8,   9],
    [  2,   2,   2,   2],
];

// ---------------------------------------------------------------------------
// Context model
// ---------------------------------------------------------------------------

/// Number of context variables (ctxIdx 0..1023, clause 9.3.1.1).
pub const NUM_CABAC_CONTEXTS: usize = 1024;

/// Adaptive probability context model for CABAC (H.264, 9.3.1).
#[derive(Debug, Clone)]
pub struct CabacContext {
    /// Probability state index (0 = equiprobable, 63 = most skewed).
    pub state: u8,
    /// Most Probable Symbol value.
    pub mps: bool,
}

/// Initialise a single CABAC context from its (m, n) pair
/// (clause 9.3.1.1 equations 9-5..9-9).
fn init_context(slice_qp: i32, m: i8, n: i8) -> CabacContext {
    let pre = ((m as i32 * slice_qp.clamp(0, 51)) >> 4) + n as i32;
    let pre = pre.clamp(1, 126);
    if pre <= 63 {
        CabacContext {
            state: (63 - pre) as u8,
            mps: false,
        }
    } else {
        CabacContext {
            state: (pre - 64) as u8,
            mps: true,
        }
    }
}

/// Initialise all context variables for a slice: the I-slice table, or the
/// `cabac_init_idc`-selected P/B table (Tables 9-12..9-33).
pub(crate) fn init_contexts(
    slice_qp: i32,
    intra_slice: bool,
    cabac_init_idc: u8,
) -> Vec<CabacContext> {
    let tab: &[(i8, i8); NUM_CABAC_CONTEXTS] = if intra_slice {
        &super::h264_cabac_init::CABAC_INIT_I
    } else {
        &super::h264_cabac_init::CABAC_INIT_PB[cabac_init_idc.min(2) as usize]
    };
    tab.iter()
        .map(|&(m, n)| init_context(slice_qp, m, n))
        .collect()
}

// ---------------------------------------------------------------------------
// CABAC arithmetic decoding engine (H.264, 9.3.3.2)
// ---------------------------------------------------------------------------

/// CABAC binary arithmetic decoder for H.264.
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    /// Next unread byte in `data`.
    offset: usize,
    /// Bit reservoir: valid bits occupy the low `bit_cnt` positions, most
    /// significant first. Renormalisation consumes several bits at once from
    /// here instead of one array access per bit.
    bit_buf: u64,
    bit_cnt: u32,
    /// Current arithmetic coding range (9-bit, initialised to 510).
    range: u32,
    /// Current arithmetic coding value.
    value: u32,
}

impl<'a> CabacDecoder<'a> {
    /// Construct a new CABAC decoder from RBSP payload bytes.
    ///
    /// The slice must start at the first CABAC-coded byte (after the slice
    /// header has been fully consumed and byte-aligned).
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = CabacDecoder {
            data,
            offset: 0,
            bit_buf: 0,
            bit_cnt: 0,
            range: 510,
            value: 0,
        };
        // Bootstrap: read 9 bits into `value` (spec 9.3.1.2).
        dec.value = dec.read_bits(9);
        dec
    }

    /// Tops up the bit reservoir from the input bytes (up to 56 valid bits).
    #[inline(always)]
    fn refill(&mut self) {
        while self.bit_cnt <= 56 && self.offset < self.data.len() {
            self.bit_buf = (self.bit_buf << 8) | u64::from(self.data[self.offset]);
            self.offset += 1;
            self.bit_cnt += 8;
        }
    }

    /// Reads `n` (≤ 9) bits, most significant first. Past the RBSP end the
    /// missing low bits read as zero (the cabac_zero_words tail).
    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.bit_cnt < n {
            self.refill();
            if self.bit_cnt < n {
                let avail = self.bit_cnt;
                let got = (self.bit_buf & ((1u64 << avail) - 1)) as u32;
                self.bit_buf = 0;
                self.bit_cnt = 0;
                return got << (n - avail);
            }
        }
        self.bit_cnt -= n;
        ((self.bit_buf >> self.bit_cnt) & ((1u64 << n) - 1)) as u32
    }

    /// Renormalization (spec 9.3.3.2.2): shift `range`/`value` left until
    /// `range >= 256`, pulling all the needed low bits in one read.
    #[inline(always)]
    fn renorm(&mut self) {
        if self.range < 256 {
            let shift = self.range.leading_zeros() - 23;
            self.range <<= shift;
            self.value = (self.value << shift) | self.read_bits(shift);
        }
    }

    /// Decode a single context-modelled binary decision (spec 9.3.3.2.1).
    #[inline(always)]
    pub fn decode_decision(&mut self, ctx: &mut CabacContext) -> bool {
        let q_idx = (self.range >> 6) & 3;
        let lps_range = RANGE_TABLE[ctx.state as usize][q_idx as usize] as u32;
        self.range -= lps_range;

        if self.value < self.range {
            // MPS path
            ctx.state = TRANSITION_MPS[ctx.state as usize];
            self.renorm();
            ctx.mps
        } else {
            // LPS path: binVal = 1 - valMPS with the *old* valMPS (spec
            // 9.3.3.2.1.1), captured before the state-0 MPS flip.
            self.value -= self.range;
            self.range = lps_range;
            let bin = !ctx.mps;
            if ctx.state == 0 {
                ctx.mps = !ctx.mps;
            }
            ctx.state = TRANSITION_LPS[ctx.state as usize];
            self.renorm();
            bin
        }
    }

    /// Decode a bypass bin (equiprobable, no context update; spec 9.3.3.2.3).
    #[inline(always)]
    pub fn decode_bypass(&mut self) -> bool {
        self.value = (self.value << 1) | self.read_bits(1);
        if self.value >= self.range {
            self.value -= self.range;
            true
        } else {
            false
        }
    }

    /// Decode a terminate bin (end_of_slice_flag / I_PCM; spec 9.3.3.2.4).
    pub fn decode_terminate(&mut self) -> bool {
        self.range -= 2;
        if self.value >= self.range {
            true
        } else {
            self.renorm();
            false
        }
    }

    /// Byte-aligns the reader for I_PCM sample data (spec 9.3.1 note): the
    /// arithmetic engine has read the 9-bit `codIOffset` window ahead of the
    /// logical position, so the aligned PCM start is 7 bits back, rounded up
    /// to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        // `offset` has read ahead into the reservoir; `bit_cnt` bits remain
        // unconsumed, so the logical bit position is `offset*8 - bit_cnt`.
        let logical = self.offset * 8 - self.bit_cnt as usize;
        let pcm_byte = logical.saturating_sub(7).div_ceil(8);
        self.offset = pcm_byte;
        self.bit_buf = 0;
        self.bit_cnt = 0;
    }

    /// Reads one raw I_PCM sample byte (after [`Self::align_to_byte`]).
    pub fn read_pcm_byte(&mut self) -> u8 {
        let b = self.data.get(self.offset).copied().unwrap_or(0);
        self.offset += 1;
        b
    }

    /// Re-initialises the arithmetic engine after the I_PCM sample data
    /// (spec 9.3.1.2), resuming from the current byte-aligned position.
    pub fn reinit_after_pcm(&mut self) {
        self.bit_buf = 0;
        self.bit_cnt = 0;
        self.range = 510;
        self.value = self.read_bits(9);
    }
}

// ---------------------------------------------------------------------------
// Syntax-element decoders (Table 9-34 binarizations + clause 9.3.3.1 contexts)
// ---------------------------------------------------------------------------

type Ctxs = [CabacContext];

/// mb_skip_flag (P slices): ctxIdx 11 + inc, where `inc` counts available
/// same-slice neighbours that are not skipped.
pub(crate) fn decode_mb_skip(dec: &mut CabacDecoder<'_>, st: &mut Ctxs, inc: usize) -> bool {
    dec.decode_decision(&mut st[11 + inc])
}

/// I-macroblock type (Table 9-36 binarization): 0 = I_NxN, 25 = I_PCM,
/// 1..=24 = I_16x16 with embedded CBP/pred-mode (the CAVLC numbering).
/// `base` = 3 with the neighbour `inc` for I slices; 17 (inc unused) for the
/// intra suffix inside P slices.
pub(crate) fn decode_intra_mb_type(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    base: usize,
    intra_slice: bool,
    inc: usize,
) -> u32 {
    let first = if intra_slice { base + inc } else { base };
    if !dec.decode_decision(&mut st[first]) {
        return 0; // I_NxN
    }
    if dec.decode_terminate() {
        return 25; // I_PCM
    }
    // I_16x16 suffix contexts: I slice → base+3..base+7; P suffix reuses
    // base+2 and base+3 for the paired bins.
    let (s1, s2a, s2b, s3a, s3b) = if intra_slice {
        (base + 3, base + 4, base + 5, base + 6, base + 7)
    } else {
        (base + 1, base + 2, base + 2, base + 3, base + 3)
    };
    let mut mb_type = 1u32;
    if dec.decode_decision(&mut st[s1]) {
        mb_type += 12; // cbp_luma = 15
    }
    if dec.decode_decision(&mut st[s2a]) {
        mb_type += 4 + 4 * dec.decode_decision(&mut st[s2b]) as u32; // cbp_chroma
    }
    mb_type += 2 * dec.decode_decision(&mut st[s3a]) as u32;
    mb_type += dec.decode_decision(&mut st[s3b]) as u32;
    mb_type
}

/// P-slice mb_type prefix: `Some(p_type)` with the CAVLC numbering
/// (0 = 16x16, 1 = 16x8, 2 = 8x16, 3 = P_8x8) or `None` when the macroblock
/// is intra (the suffix follows via [`decode_intra_mb_type`] at base 17).
pub(crate) fn decode_p_mb_type(dec: &mut CabacDecoder<'_>, st: &mut Ctxs) -> Option<u32> {
    if dec.decode_decision(&mut st[14]) {
        return None;
    }
    Some(if !dec.decode_decision(&mut st[15]) {
        3 * dec.decode_decision(&mut st[16]) as u32
    } else {
        2 - dec.decode_decision(&mut st[17]) as u32
    })
}

/// P-slice sub_mb_type (Table 9-38): the CAVLC numbering
/// (0 = 8x8, 1 = 8x4, 2 = 4x8, 3 = 4x4).
pub(crate) fn decode_sub_mb_type_p(dec: &mut CabacDecoder<'_>, st: &mut Ctxs) -> u32 {
    if dec.decode_decision(&mut st[21]) {
        return 0;
    }
    if !dec.decode_decision(&mut st[22]) {
        return 1;
    }
    if dec.decode_decision(&mut st[23]) {
        2
    } else {
        3
    }
}

/// mb_skip_flag for B slices (ctxIdxOffset 24; P uses [`decode_mb_skip`] at 11).
pub(crate) fn decode_mb_skip_b(dec: &mut CabacDecoder<'_>, st: &mut Ctxs, inc: usize) -> bool {
    dec.decode_decision(&mut st[24 + inc])
}

/// B-slice mb_type prefix (Table 9-37, ctxIdxOffset 27). Returns the Table 7-14
/// value `0..=22` for an inter B macroblock, or `None` when the macroblock is
/// intra (its suffix follows via [`decode_intra_mb_type`] at base 32). `inc` is
/// the binIdx-0 context increment (neighbours that are neither B_Skip nor
/// B_Direct_16x16, clause 9.3.3.1.1.3).
pub(crate) fn decode_b_mb_type(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    inc: usize,
) -> Option<u32> {
    if !dec.decode_decision(&mut st[27 + inc]) {
        return Some(0); // B_Direct_16x16
    }
    if !dec.decode_decision(&mut st[27 + 3]) {
        // B_L0_16x16 / B_L1_16x16
        return Some(1 + dec.decode_decision(&mut st[27 + 5]) as u32);
    }
    let mut bits = (dec.decode_decision(&mut st[27 + 4]) as u32) << 3;
    bits += (dec.decode_decision(&mut st[27 + 5]) as u32) << 2;
    bits += (dec.decode_decision(&mut st[27 + 5]) as u32) << 1;
    bits += dec.decode_decision(&mut st[27 + 5]) as u32;
    if bits < 8 {
        return Some(bits + 3); // B_Bi_16x16 .. B_L1_L0_16x8
    }
    if bits == 13 {
        return None; // intra
    }
    if bits == 14 {
        return Some(11); // B_L1_L0_8x16
    }
    if bits == 15 {
        return Some(22); // B_8x8
    }
    bits = (bits << 1) + dec.decode_decision(&mut st[27 + 5]) as u32;
    Some(bits - 4) // B_L0_Bi_16x8 .. B_Bi_Bi_8x16
}

/// B-slice sub_mb_type (Table 9-38, ctxIdxOffset 36). Returns the Table 7-18
/// value `0..=12` (0 = B_Direct_8x8, 1/2 = L0/L1 8x8, 3 = Bi 8x8, 4..9 = 8x4/4x8
/// halves, 10..12 = 4x4 quarters).
pub(crate) fn decode_sub_mb_type_b(dec: &mut CabacDecoder<'_>, st: &mut Ctxs) -> u32 {
    if !dec.decode_decision(&mut st[36]) {
        return 0; // B_Direct_8x8
    }
    if !dec.decode_decision(&mut st[37]) {
        return 1 + dec.decode_decision(&mut st[39]) as u32; // B_L0_8x8 / B_L1_8x8
    }
    let mut ty = 3u32;
    if dec.decode_decision(&mut st[38]) {
        if dec.decode_decision(&mut st[39]) {
            return 11 + dec.decode_decision(&mut st[39]) as u32; // B_L1_4x4 / B_Bi_4x4
        }
        ty += 4;
    }
    ty += 2 * dec.decode_decision(&mut st[39]) as u32;
    ty += dec.decode_decision(&mut st[39]) as u32;
    ty
}

/// prev_intra4x4_pred_mode_flag + rem_intra4x4_pred_mode (ctx 68/69; the
/// three rem bins are least-significant first).
pub(crate) fn decode_intra4x4_pred_mode(dec: &mut CabacDecoder<'_>, st: &mut Ctxs, pred: u8) -> u8 {
    if dec.decode_decision(&mut st[68]) {
        return pred;
    }
    let mut mode = dec.decode_decision(&mut st[69]) as u8;
    mode += 2 * dec.decode_decision(&mut st[69]) as u8;
    mode += 4 * dec.decode_decision(&mut st[69]) as u8;
    mode + (mode >= pred) as u8
}

/// intra_chroma_pred_mode: ctx 64 + inc (neighbours with nonzero mode), then
/// truncated unary with ctx 67.
pub(crate) fn decode_chroma_pred_mode(dec: &mut CabacDecoder<'_>, st: &mut Ctxs, inc: usize) -> u8 {
    if !dec.decode_decision(&mut st[64 + inc]) {
        return 0;
    }
    if !dec.decode_decision(&mut st[67]) {
        return 1;
    }
    if !dec.decode_decision(&mut st[67]) {
        2
    } else {
        3
    }
}

/// coded_block_pattern luma bits (clause 9.3.3.1.1.4): each bin's context
/// looks at the co-located 8x8 CBP bit of the neighbour on that side
/// (`cbp_a` left, `cbp_b` top; the caller substitutes the availability
/// defaults).
pub(crate) fn decode_cbp_luma(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    cbp_a: u32,
    cbp_b: u32,
) -> u32 {
    let mut cbp = 0u32;
    let ctx = ((cbp_a & 0x02) == 0) as usize + 2 * ((cbp_b & 0x04) == 0) as usize;
    cbp |= dec.decode_decision(&mut st[73 + ctx]) as u32;
    let ctx = ((cbp & 0x01) == 0) as usize + 2 * ((cbp_b & 0x08) == 0) as usize;
    cbp |= (dec.decode_decision(&mut st[73 + ctx]) as u32) << 1;
    let ctx = ((cbp_a & 0x08) == 0) as usize + 2 * ((cbp & 0x01) == 0) as usize;
    cbp |= (dec.decode_decision(&mut st[73 + ctx]) as u32) << 2;
    let ctx = ((cbp & 0x04) == 0) as usize + 2 * ((cbp & 0x02) == 0) as usize;
    cbp | ((dec.decode_decision(&mut st[73 + ctx]) as u32) << 3)
}

/// coded_block_pattern chroma (0/1/2): ctx 77 + inc from the neighbours'
/// chroma CBP values.
pub(crate) fn decode_cbp_chroma(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    cbp_a: u32,
    cbp_b: u32,
) -> u32 {
    let inc = (cbp_a > 0) as usize + 2 * (cbp_b > 0) as usize;
    if !dec.decode_decision(&mut st[77 + inc]) {
        return 0;
    }
    let inc = 4 + (cbp_a == 2) as usize + 2 * (cbp_b == 2) as usize;
    1 + dec.decode_decision(&mut st[77 + inc]) as u32
}

/// mb_qp_delta: unary bins at ctx 60 + (previous delta != 0), 62, 63...,
/// mapped to a signed delta (Table 9-3).
pub(crate) fn decode_mb_qp_delta(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    prev_nonzero: bool,
) -> i32 {
    if !dec.decode_decision(&mut st[60 + prev_nonzero as usize]) {
        return 0;
    }
    let mut val = 1i32;
    let mut ctx = 62usize;
    while dec.decode_decision(&mut st[ctx]) {
        ctx = 63;
        val += 1;
        if val > 102 {
            break; // corrupt stream guard
        }
    }
    if val & 1 == 1 {
        (val + 1) >> 1
    } else {
        -((val + 1) >> 1)
    }
}

/// ref_idx_l0: unary at ctx 54 + inc (neighbour partitions with refIdx > 0),
/// continuing at 58 then 59.
pub(crate) fn decode_ref_idx(dec: &mut CabacDecoder<'_>, st: &mut Ctxs, inc: usize) -> u32 {
    let mut ctx = inc;
    let mut r = 0u32;
    while dec.decode_decision(&mut st[54 + ctx]) {
        ctx = (ctx >> 2) + 4;
        r += 1;
        if r >= 32 {
            break; // corrupt stream guard
        }
    }
    r
}

/// One mvd component (UEG3 binarization, clause 9.3.2.3): `base` is 40 for
/// mvd_x, 47 for mvd_y; `amvd` = |mvd_A| + |mvd_B| of the same component.
/// Returns the signed mvd and its magnitude capped at 70 for the neighbour
/// cache (the cap cannot change future threshold tests).
pub(crate) fn decode_mvd(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    base: usize,
    amvd: u32,
) -> (i32, u8) {
    let inc = (amvd > 2) as usize + (amvd > 32) as usize;
    if !dec.decode_decision(&mut st[base + inc]) {
        return (0, 0);
    }
    let mut mvd = 1u32;
    let mut ctx = base + 3;
    while mvd < 9 && dec.decode_decision(&mut st[ctx]) {
        if mvd < 4 {
            ctx += 1;
        }
        mvd += 1;
    }
    if mvd >= 9 {
        // Exp-Golomb (k = 3) escape via bypass bins.
        let mut k = 3u32;
        while dec.decode_bypass() {
            mvd += 1 << k;
            k += 1;
            if k > 24 {
                break; // corrupt stream guard
            }
        }
        while k > 0 {
            k -= 1;
            mvd += (dec.decode_bypass() as u32) << k;
        }
    }
    let stored = mvd.min(70) as u8;
    let signed = if dec.decode_bypass() {
        -(mvd as i32)
    } else {
        mvd as i32
    };
    (signed, stored)
}

/// coded_block_flag for block category `cat` (0 luma DC, 1 luma AC, 2 luma
/// 4x4, 3 chroma DC, 4 chroma AC) with the neighbour-derived `inc`.
pub(crate) fn decode_cbf(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    cat: usize,
    inc: usize,
) -> bool {
    const BASE: [usize; 5] = [85, 89, 93, 97, 101];
    dec.decode_decision(&mut st[BASE[cat] + inc])
}

/// Residual levels after a set coded_block_flag (clause 9.3.2.7): the
/// significance map, then coeff_abs_level_minus1 + signs in reverse scan
/// order. Fills `out[pos]` at scan positions `[0, max_coeff)` and returns the
/// number of nonzero coefficients.
pub(crate) fn decode_residual_levels(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    cat: usize,
    max_coeff: usize,
    out: &mut [i32; 16],
) -> usize {
    const SIG_BASE: [usize; 5] = [105, 105 + 15, 105 + 29, 105 + 44, 105 + 47];
    const LAST_BASE: [usize; 5] = [166, 166 + 15, 166 + 29, 166 + 44, 166 + 47];
    const ABS_BASE: [usize; 5] = [227, 227 + 10, 227 + 20, 227 + 30, 227 + 39];
    // Level-context node machine (clause 9.3.3.1.3): nodes 0..3 count
    // trailing ones, 4..7 mark a level > 1 already seen.
    const LEVEL1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
    const LEVELGT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
    const TRANS_EQ1: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
    const TRANS_GT1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

    let (sig_base, last_base, abs_base) = (SIG_BASE[cat], LAST_BASE[cat], ABS_BASE[cat]);
    let mut index = [0usize; 16];
    let mut count = 0usize;
    let mut pos = 0usize;
    while pos < max_coeff - 1 {
        if dec.decode_decision(&mut st[sig_base + pos]) {
            index[count] = pos;
            count += 1;
            if dec.decode_decision(&mut st[last_base + pos]) {
                pos = max_coeff;
                break;
            }
        }
        pos += 1;
    }
    if pos == max_coeff - 1 {
        // Ran through every earlier position: the final one is significant.
        index[count] = pos;
        count += 1;
    }

    let mut node = 0usize;
    for k in (0..count).rev() {
        let p = index[k];
        if !dec.decode_decision(&mut st[abs_base + LEVEL1_CTX[node]]) {
            node = TRANS_EQ1[node];
            out[p] = if dec.decode_bypass() { -1 } else { 1 };
        } else {
            let mut abs = 2u32;
            let gt1 = abs_base + LEVELGT1_CTX[node];
            node = TRANS_GT1[node];
            while abs < 15 && dec.decode_decision(&mut st[gt1]) {
                abs += 1;
            }
            if abs >= 15 {
                // Exp-Golomb (k = 0) escape via bypass bins.
                let mut j = 0u32;
                while dec.decode_bypass() && j < 23 {
                    j += 1;
                }
                abs = 1;
                while j > 0 {
                    j -= 1;
                    abs = abs * 2 + dec.decode_bypass() as u32;
                }
                abs += 14;
            }
            out[p] = if dec.decode_bypass() {
                -(abs as i32)
            } else {
                abs as i32
            };
        }
    }
    count
}

/// transform_size_8x8_flag (clause 9.3.3.1.1.10): ctxIdxOffset 399, with
/// ctxIdxInc = whether the left / top macroblock used the 8x8 transform.
pub(crate) fn decode_transform_size_8x8_flag(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    inc: usize,
) -> bool {
    dec.decode_decision(&mut st[399 + inc])
}

/// Luma 8x8 residual (ctxBlockCat 5) for the frame-coded case: the 63-position
/// significance map uses the position→context maps of Table 9-43 rather than
/// the raw scan position, then the same reverse-scan level machine as the 4x4
/// path. Fills `out` at scan positions `[0, 64)` and returns the nonzero count.
/// In 4:2:0 the 8x8 luma block carries no coded_block_flag — the caller decodes
/// this only when the CBP marks the 8x8 as coded.
pub(crate) fn decode_residual_8x8(
    dec: &mut CabacDecoder<'_>,
    st: &mut Ctxs,
    out: &mut [i32; 64],
) -> usize {
    // Table 9-43 (frame): significant_coeff_flag / last_significant_coeff_flag
    // ctxIdxInc as a function of the scan position.
    const SIG_MAP: [u8; 63] = [
        0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9,
        8, 7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9,
        11, 12, 13, 11, 14, 10, 12,
    ];
    const LAST_MAP: [u8; 63] = [
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
        8, 8, 8,
    ];
    const SIG_BASE: usize = 402;
    const LAST_BASE: usize = 417;
    const ABS_BASE: usize = 426;
    const LEVEL1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
    const LEVELGT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
    const TRANS_EQ1: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
    const TRANS_GT1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

    let mut index = [0usize; 64];
    let mut count = 0usize;
    let mut pos = 0usize;
    while pos < 63 {
        if dec.decode_decision(&mut st[SIG_BASE + SIG_MAP[pos] as usize]) {
            index[count] = pos;
            count += 1;
            if dec.decode_decision(&mut st[LAST_BASE + LAST_MAP[pos] as usize]) {
                pos = 64;
                break;
            }
        }
        pos += 1;
    }
    if pos == 63 {
        index[count] = pos;
        count += 1;
    }

    let mut node = 0usize;
    for k in (0..count).rev() {
        let p = index[k];
        if !dec.decode_decision(&mut st[ABS_BASE + LEVEL1_CTX[node]]) {
            node = TRANS_EQ1[node];
            out[p] = if dec.decode_bypass() { -1 } else { 1 };
        } else {
            let mut abs = 2u32;
            let gt1 = ABS_BASE + LEVELGT1_CTX[node];
            node = TRANS_GT1[node];
            while abs < 15 && dec.decode_decision(&mut st[gt1]) {
                abs += 1;
            }
            if abs >= 15 {
                let mut j = 0u32;
                while dec.decode_bypass() && j < 23 {
                    j += 1;
                }
                abs = 1;
                while j > 0 {
                    j -= 1;
                    abs = abs * 2 + dec.decode_bypass() as u32;
                }
                abs += 14;
            }
            out[p] = if dec.decode_bypass() {
                -(abs as i32)
            } else {
                abs as i32
            };
        }
    }
    count
}

/// Identifies the entropy coding mode from a PPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCodingMode {
    /// Context-Adaptive Variable-Length Coding (Baseline).
    Cavlc,
    /// Context-Adaptive Binary Arithmetic Coding (Main/High).
    Cabac,
}

impl EntropyCodingMode {
    /// Determine entropy coding mode from `entropy_coding_mode_flag`.
    pub fn from_flag(flag: bool) -> Self {
        if flag {
            EntropyCodingMode::Cabac
        } else {
            EntropyCodingMode::Cavlc
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_init_matches_spec_formula() {
        // ctx 3 in the I table is (m, n) = (20, -15): at QP 26,
        // pre = ((20 * 26) >> 4) - 15 = 32 - 15 = 17 → state 63-17=46, mps 0.
        let st = init_contexts(26, true, 0);
        assert_eq!(st.len(), NUM_CABAC_CONTEXTS);
        assert_eq!(st[3].state, 63 - 17);
        assert!(!st[3].mps);
    }

    #[test]
    fn pb_tables_select_by_init_idc() {
        let a = init_contexts(30, false, 0);
        let b = init_contexts(30, false, 1);
        // The three P/B variants differ somewhere in the mvd contexts.
        assert!((40..54).any(|i| a[i].state != b[i].state || a[i].mps != b[i].mps));
    }

    #[test]
    fn bypass_deterministic_on_zeros() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut dec = CabacDecoder::new(&data);
        for _ in 0..8 {
            assert!(!dec.decode_bypass());
        }
    }

    #[test]
    fn terminate_fires_on_all_ones() {
        // value after init = 0x1FF = 511; range 510 - 2 = 508; 511 >= 508.
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut dec = CabacDecoder::new(&data);
        assert!(dec.decode_terminate());
    }

    #[test]
    fn decision_updates_state() {
        let data = [0x5A, 0xC3, 0x99, 0x11, 0x22];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = CabacContext {
            state: 10,
            mps: false,
        };
        let before = ctx.state;
        let _ = dec.decode_decision(&mut ctx);
        assert_ne!(ctx.state, before);
    }

    #[test]
    fn transition_tables_bounded() {
        for s in 0..64 {
            assert!(TRANSITION_MPS[s] < 64);
            assert!(TRANSITION_LPS[s] < 64);
            for q in 0..4 {
                assert!(RANGE_TABLE[s][q] >= 2);
            }
        }
    }
}
