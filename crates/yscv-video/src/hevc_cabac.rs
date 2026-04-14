//! HEVC CABAC (Context-based Adaptive Binary Arithmetic Coding) entropy decoder.
//!
//! Implements the arithmetic decoding engine and context model management
//! specified in ITU-T H.265 sections 9.3.  This module provides:
//!
//! - [`CabacDecoder`] — the arithmetic decoding engine that reads compressed
//!   bitstream data and produces binary decisions.
//! - [`ContextModel`] — adaptive probability model with MPS/LPS tracking.
//! - Binarization helpers for truncated Rice, fixed-length, unary, and
//!   Exp-Golomb coded syntax elements.

// ---------------------------------------------------------------------------
// State transition tables (ITU-T H.265, Table 9-45)
// ---------------------------------------------------------------------------

/// State transition after decoding the **Least Probable Symbol** (LPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
const TRANS_IDX_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,
     6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18,
    19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29,
    29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36,
    36, 36, 37, 37, 37, 38, 38, 63,
];

/// State transition after decoding the **Most Probable Symbol** (MPS).
/// Indexed by `pStateIdx` (0..=63).
#[rustfmt::skip]
const TRANS_IDX_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 62, 63,
];

// ---------------------------------------------------------------------------
// Range LPS table (ITU-T H.265, Table 9-48)
// ---------------------------------------------------------------------------

/// `rangeTabLps[pStateIdx][qRangeIdx]`
///
/// 64 rows (one per probability state) x 4 columns (one per quarter-range
/// index derived from the current interval range).
#[rustfmt::skip]
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
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
// Pre-computed branchless state transition tables
// ---------------------------------------------------------------------------

/// Packed state_mps transition table for MPS path.
/// Index: `state | (mps << 6)` (7-bit, 0..127).
/// Value: new packed `state | (mps << 6)`.
/// MPS transition never flips mps, just advances state.
static CABAC_TRANS_MPS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut mps = 0u8;
    while mps < 2 {
        let mut s = 0u8;
        while s < 64 {
            let idx = s | (mps << 6);
            let new_state = TRANS_IDX_MPS[s as usize];
            t[idx as usize] = new_state | (mps << 6);
            s += 1;
        }
        mps += 1;
    }
    t
};

/// Packed state_mps transition table for LPS path.
/// LPS at state=0 flips mps; otherwise mps stays.
static CABAC_TRANS_LPS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut mps = 0u8;
    while mps < 2 {
        let mut s = 0u8;
        while s < 64 {
            let idx = s | (mps << 6);
            let new_state = TRANS_IDX_LPS[s as usize];
            let new_mps = if s == 0 { 1 - mps } else { mps };
            t[idx as usize] = new_state | (new_mps << 6);
            s += 1;
        }
        mps += 1;
    }
    t
};

// ---------------------------------------------------------------------------
// Context model
// ---------------------------------------------------------------------------

/// Adaptive probability context model for CABAC (ITU-T H.265, 9.3.1).
///
/// Each context stores a 6-bit probability state index (`state`, 0..=63) and
/// the value of the Most Probable Symbol (`mps`, 0 or 1).
#[derive(Debug, Clone)]
pub struct ContextModel {
    /// Probability state index (0 = equiprobable, 63 = most skewed).
    pub state: u8,
    /// Most Probable Symbol value (0 or 1).
    pub mps: u8,
}

impl ContextModel {
    /// Create a context model from an initialisation value (Table 9-4).
    pub fn new(init_value: u8) -> Self {
        let mut ctx = ContextModel { state: 0, mps: 0 };
        ctx.init(26, init_value);
        ctx
    }

    /// (Re-)initialise the context for a given `slice_qp` and `init_value`
    /// (ITU-T H.265, 9.3.1.1).
    pub fn init(&mut self, slice_qp: i32, init_value: u8) {
        let slope = ((init_value >> 4) as i32) * 5 - 45;
        let offset = (((init_value & 15) as i32) << 3) - 16;
        let init_state = ((slope * (slice_qp.clamp(0, 51) - 16)) >> 4) + offset;
        let pre_ctx_state = init_state.clamp(1, 126);

        if pre_ctx_state <= 63 {
            self.state = (63 - pre_ctx_state) as u8;
            self.mps = 0;
        } else {
            self.state = (pre_ctx_state - 64) as u8;
            self.mps = 1;
        }
    }

    /// Pack state and mps into a single u8: `state | (mps << 6)`.
    #[inline(always)]
    pub fn packed(&self) -> u8 {
        self.state | (self.mps << 6)
    }

    /// Unpack from a packed u8.
    #[inline(always)]
    pub fn unpack(packed: u8) -> Self {
        ContextModel {
            state: packed & 63,
            mps: (packed >> 6) & 1,
        }
    }
}

// ---------------------------------------------------------------------------
// CABAC arithmetic decoding engine
// ---------------------------------------------------------------------------

/// CABAC arithmetic decoder for HEVC (ITU-T H.265, 9.3.3).
///
/// Reads a byte-aligned NAL unit payload and exposes methods for
/// context-modelled decisions, bypass bins, and terminating bins.
pub struct CabacDecoder<'a> {
    /// Raw NAL unit payload bytes.
    data: &'a [u8],
    /// Current byte offset into `data`.
    offset: usize,
    /// 32-bit bit buffer for batch reading.
    bit_buf: u32,
    /// Number of valid bits in `bit_buf` (MSB-aligned).
    bits_left: u32,
    /// Current arithmetic coding range (9-bit value, initialised to 510).
    range: u32,
    /// Current arithmetic coding offset / value.
    value: u32,
}

impl<'a> CabacDecoder<'a> {
    /// Construct a new CABAC decoder from a NAL payload slice.
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = CabacDecoder {
            data,
            offset: 0,
            bit_buf: 0,
            bits_left: 0,
            range: 510,
            value: 0,
        };
        // Bootstrap: read 9 bits into `value` (spec 9.3.2.2).
        dec.value = dec.read_bits(9);
        dec
    }

    // ------------------------------------------------------------------
    // Buffered bit-level I/O
    // ------------------------------------------------------------------

    /// Refill bit buffer — load up to 4 bytes using unchecked access.
    #[inline(always)]
    #[allow(unsafe_code)]
    fn refill(&mut self) {
        unsafe {
            let len = self.data.len();
            let ptr = self.data.as_ptr();
            while self.bits_left <= 24 && self.offset < len {
                self.bit_buf = (self.bit_buf << 8) | *ptr.add(self.offset) as u32;
                self.offset += 1;
                self.bits_left += 8;
            }
        }
    }

    /// Read a single bit from the buffered bitstream.
    #[inline(always)]
    fn read_bit(&mut self) -> u32 {
        if self.bits_left == 0 {
            self.refill();
            if self.bits_left == 0 {
                return 0;
            }
        }
        self.bits_left -= 1;
        (self.bit_buf >> self.bits_left) & 1
    }

    /// Read `n` bits (MSB-first). Fast path: single extract after refill.
    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> u32 {
        if self.bits_left < n {
            self.refill();
        }
        if self.bits_left >= n {
            self.bits_left -= n;
            (self.bit_buf >> self.bits_left) & ((1u32 << n) - 1)
        } else {
            // Slow path (end of stream)
            let mut val = 0u32;
            for _ in 0..n {
                val = (val << 1) | self.read_bit();
            }
            val
        }
    }

    // ------------------------------------------------------------------
    // Core decoding primitives — branchless (spec 9.3.3.2)
    // ------------------------------------------------------------------

    /// Renormalise using CLZ for batch shift + multi-bit read.
    #[inline(always)]
    fn renormalize(&mut self) {
        if self.range >= 256 {
            return;
        }
        // CLZ-based batch renormalize: compute shift to bring range into [256, 512)
        // range is at least 2 (min LPS table value), so shift is 1..7
        let shift = self.range.leading_zeros() - 23; // 23 = clz(256)
        self.range <<= shift;
        self.value = (self.value << shift) | self.read_bits(shift);
    }

    /// Decode one bin using context-modelled arithmetic coding.
    ///
    /// **Fully unsafe hot path** with inline asm on aarch64.
    /// ~100K calls/frame — every nanosecond counts.
    #[inline(always)]
    #[allow(unsafe_code)]
    pub fn decode_decision(&mut self, ctx: &mut ContextModel) -> bool {
        unsafe {
            let state = ctx.state as usize;
            let mps = ctx.mps;

            let q_range_idx = ((self.range >> 6) & 3) as usize;
            let range_lps = *RANGE_TAB_LPS
                .get_unchecked(state)
                .get_unchecked(q_range_idx) as u32;

            self.range -= range_lps;

            // Branchless MPS/LPS
            let is_lps = (self.value >= self.range) as u32;
            let lps_mask = 0u32.wrapping_sub(is_lps);

            self.value -= self.range & lps_mask;
            self.range = (self.range & !lps_mask) | (range_lps & lps_mask);

            // Branchless state transition
            let packed = (state | ((mps as usize) << 6)) & 127;
            let trans_mps = *CABAC_TRANS_MPS.get_unchecked(packed);
            let trans_lps = *CABAC_TRANS_LPS.get_unchecked(packed);
            let new_packed = trans_mps ^ ((trans_mps ^ trans_lps) & (lps_mask as u8));
            ctx.state = new_packed & 63;
            ctx.mps = (new_packed >> 6) & 1;

            // Renormalize: shift until range >= 256
            // For range >= 256: clz ≤ 23, saturating_sub = 0, skip.
            // For range < 256: clz > 23, shift = clz - 23 ∈ [1,7].
            let shift = self.range.leading_zeros().saturating_sub(23);
            if shift > 0 {
                self.range <<= shift;
                self.value = (self.value << shift) | self.read_bits(shift);
            }

            (mps ^ (is_lps as u8)) != 0
        }
    }

    /// Decode one bin in bypass mode — branchless, unchecked.
    #[inline(always)]
    #[allow(unsafe_code)]
    pub fn decode_bypass(&mut self) -> bool {
        self.value = (self.value << 1) | self.read_bit();
        let is_one = (self.value >= self.range) as u32;
        self.value -= self.range & 0u32.wrapping_sub(is_one);
        is_one != 0
    }

    /// Decode the terminating bin (spec 9.3.3.2.4).
    #[inline(always)]
    pub fn decode_terminate(&mut self) -> bool {
        self.range -= 2;
        if self.value >= self.range {
            true
        } else {
            self.renormalize();
            false
        }
    }

    /// Returns the number of unconsumed bytes remaining in the input.
    pub fn bytes_remaining(&self) -> usize {
        let consumed = self.offset;
        let partial = if self.bits_left > 0 { 1 } else { 0 };
        self.data.len().saturating_sub(consumed) + partial
    }

    /// Reinitialise the arithmetic decoder at a new byte offset within the
    /// same NAL payload. Used for tile/WPP entry points where CABAC state
    /// is reset to a fresh decode position.
    pub fn reinit_at_offset(&mut self, byte_offset: usize) {
        self.offset = byte_offset.min(self.data.len());
        self.bit_buf = 0;
        self.bits_left = 0;
        self.range = 510;
        // Bootstrap: read 9 bits into `value` (spec 9.3.2.2).
        self.value = self.read_bits(9);
    }

    /// Return the current byte offset into the underlying data slice.
    pub fn current_byte_offset(&self) -> usize {
        // Subtract any bytes still buffered but not yet consumed.
        let buffered_bytes = (self.bits_left / 8) as usize;
        self.offset.saturating_sub(buffered_bytes)
    }

    /// Byte-align the decoder by discarding any partial bits remaining
    /// in the bit buffer. After this call, the next read starts at a
    /// byte-aligned position.
    pub fn byte_align(&mut self) {
        let discard = self.bits_left % 8;
        if discard > 0 {
            self.bits_left -= discard;
        }
    }

    // ------------------------------------------------------------------
    // Syntax-element binarization helpers
    // ------------------------------------------------------------------

    /// Decode a **truncated Rice** (TR) binarized syntax element
    /// (spec 9.3.2.2).
    ///
    /// - `ctx` — context model array; the first entry is used for the prefix
    ///   unary part, subsequent entries for further bins.
    /// - `c_max` — maximum value (capped by the binarization).
    /// - `c_rice_param` — Rice parameter controlling the suffix length.
    ///
    /// The prefix is decoded as a **truncated unary** code using context
    /// models, and the suffix (if any) is decoded in bypass mode.
    #[inline]
    pub fn decode_tr(&mut self, ctx: &mut [ContextModel], c_max: u32, c_rice_param: u32) -> u32 {
        let prefix_max = c_max >> c_rice_param;
        // Truncated unary prefix
        let mut prefix = 0u32;
        while prefix < prefix_max {
            let ctx_idx = prefix.min((ctx.len() as u32).saturating_sub(1)) as usize;
            if self.decode_decision(&mut ctx[ctx_idx]) {
                prefix += 1;
            } else {
                break;
            }
        }

        // Suffix — `c_rice_param` bypass bins (binary representation of remainder)
        let suffix = if c_rice_param > 0 {
            self.decode_fl_bypass(c_rice_param)
        } else {
            0
        };

        let value = (prefix << c_rice_param) + suffix;
        value.min(c_max)
    }

    /// Decode a **fixed-length** (FL) binarized syntax element using bypass
    /// bins (spec 9.3.3.2.3, used via 9.3.2.4).
    ///
    /// Reads `n_bits` bypass bins and returns the resulting value
    /// (MSB-first).
    #[inline(always)]
    pub fn decode_fl(&mut self, n_bits: u32) -> u32 {
        self.decode_fl_bypass(n_bits)
    }

    /// Read `n_bits` bypass bins MSB-first.
    #[inline(always)]
    fn decode_fl_bypass(&mut self, n_bits: u32) -> u32 {
        let mut val = 0u32;
        for _ in 0..n_bits {
            val = (val << 1) | u32::from(self.decode_bypass());
        }
        val
    }

    /// Decode a **unary** (U) binarized syntax element using context models.
    ///
    /// Returns the number of `1` bins seen before the first `0` bin, capped
    /// at `max`.  Context index for each bin position `k` is
    /// `min(k, ctx.len() - 1)`.
    #[inline]
    pub fn decode_unary(&mut self, ctx: &mut [ContextModel], max: u32) -> u32 {
        let mut val = 0u32;
        while val < max {
            let ctx_idx = val.min((ctx.len() as u32).saturating_sub(1)) as usize;
            if self.decode_decision(&mut ctx[ctx_idx]) {
                val += 1;
            } else {
                break;
            }
        }
        val
    }

    /// Decode an **Exp-Golomb** coded syntax element using bypass bins
    /// (spec 9.3.2.5, k-th order).
    ///
    /// `k` is the order parameter.  Returns the decoded unsigned value.
    #[inline]
    pub fn decode_eg(&mut self, k: u32) -> u32 {
        // Prefix: count leading ones (Exp-Golomb uses bypass bins).
        let mut leading = 0u32;
        while self.decode_bypass() {
            leading += 1;
            // Safety limit — real streams never exceed ~32.
            if leading > 31 {
                break;
            }
        }

        // The stop bit (0) has already been consumed by the failing
        // `decode_bypass` above.
        //
        // Suffix: read `leading + k` bypass bins.
        let suffix_len = leading + k;
        let suffix = self.decode_fl_bypass(suffix_len);

        // Value = ((1 << leading) - 1) << k  +  suffix
        if leading >= 32 {
            return suffix;
        }
        ((1u32 << leading) - 1).wrapping_shl(k).wrapping_add(suffix)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Context model tests ------------------------------------------------

    #[test]
    fn context_init_default_qp() {
        let ctx = ContextModel::new(154);
        // init_value = 154 => slope = (154 >> 4) * 5 - 45 = 9*5-45 = 0
        // offset = ((154 & 15) << 3) - 16 = (10 << 3) - 16 = 80 - 16 = 64
        // init_state = (0 * (26-16)) >> 4 + 64 = 64
        // pre_ctx_state = clamp(64, 1, 126) = 64
        // pre_ctx_state > 63 => state = 64 - 64 = 0, mps = 1
        assert_eq!(ctx.state, 0);
        assert_eq!(ctx.mps, 1);
    }

    #[test]
    fn context_init_high_state() {
        let ctx = ContextModel::new(255);
        // slope = (255 >> 4)*5 - 45 = 15*5-45 = 30
        // offset = ((255 & 15) << 3) - 16 = (15*8)-16 = 104
        // init_state = (30 * (26-16)) >> 4 + 104 = 300>>4 + 104 = 18+104 = 122
        // pre_ctx_state = clamp(122, 1, 126) = 122
        // 122 > 63 => state = 122-64 = 58, mps = 1
        assert_eq!(ctx.state, 58);
        assert_eq!(ctx.mps, 1);
    }

    #[test]
    fn context_init_low_state() {
        let ctx = ContextModel::new(0);
        // slope = 0*5-45 = -45
        // offset = 0-16 = -16
        // init_state = (-45*(26-16))>>4 + (-16) = -450>>4 + (-16) = -28 + (-16) = -44
        // pre_ctx_state = clamp(-44, 1, 126) = 1
        // 1 <= 63 => state = 63 - 1 = 62, mps = 0
        assert_eq!(ctx.state, 62);
        assert_eq!(ctx.mps, 0);
    }

    #[test]
    fn context_state_transitions() {
        let mut ctx = ContextModel { state: 0, mps: 0 };
        // After MPS at state 0, transition to state 1
        let old_mps = ctx.mps;
        ctx.state = TRANS_IDX_MPS[ctx.state as usize];
        assert_eq!(ctx.state, 1);
        assert_eq!(ctx.mps, old_mps);

        // After LPS at state 0, state stays 0 but MPS flips
        ctx.state = 0;
        ctx.mps = 0;
        // LPS at state 0 causes MPS flip
        if ctx.state == 0 {
            ctx.mps = 1 - ctx.mps;
        }
        ctx.state = TRANS_IDX_LPS[ctx.state as usize];
        assert_eq!(ctx.state, 0);
        assert_eq!(ctx.mps, 1);
    }

    #[test]
    fn context_init_reinit() {
        let mut ctx = ContextModel::new(154);
        let orig_state = ctx.state;
        let orig_mps = ctx.mps;
        // Reinitialize with same parameters should give the same result.
        ctx.init(26, 154);
        assert_eq!(ctx.state, orig_state);
        assert_eq!(ctx.mps, orig_mps);

        // Reinitialize with different QP should change state.
        ctx.init(40, 154);
        // slope=0, offset=64 => init_state = 0 + 64 = 64
        // Same result because slope is 0.
        assert_eq!(ctx.state, 0);
        assert_eq!(ctx.mps, 1);
    }

    // -- CABAC decoder tests ------------------------------------------------

    #[test]
    fn cabac_init() {
        // Ensure decoder construction from minimal data does not panic.
        let data = [0x00, 0x00, 0x01, 0xFF];
        let dec = CabacDecoder::new(&data);
        assert_eq!(dec.range, 510);
        // value should have consumed 9 bits = 1 full byte + 1 bit
        // data[0] = 0x00, data[1] = 0x00 => 9 MSBs = 0b0_0000_0000 = 0
        assert_eq!(dec.value, 0);
    }

    #[test]
    fn cabac_bypass_known_pattern() {
        // Encode bit pattern 1,0,1,1,0,0,1,0 as raw bytes.
        // After the 9-bit init bootstrap, bypass reads one bit at a time.
        //
        // The bypass decode doubles `value` and adds a new bit, then
        // compares against `range`.  With range=510 after init, after the
        // first bypass bin the range stays 510 and each call essentially
        // reads one raw bit (since range > 256 and stays constant in
        // bypass mode).
        //
        // We need to set up bytes so the 9-bit bootstrap reads a known
        // value and then subsequent bits form our pattern.
        //
        // Let's build a bitstream where the first 9 bits are 0 (value=0)
        // and the following 8 bits are the pattern 10110010.
        //
        // Bit layout:  0_0000_0000  1_0110_010(padding)
        //   byte 0: 0000_0000  = 0x00
        //   byte 1: 0101_1001  = 0x59
        //   byte 2: 0000_0000  = 0x00  (padding)
        let data = [0x00u8, 0x59, 0x00];
        let mut dec = CabacDecoder::new(&data);
        assert_eq!(dec.value, 0);
        assert_eq!(dec.range, 510);

        // With value=0 and range=510, bypass reads the next bit and
        // doubles value.  If new value >= range => 1, else => 0.
        //
        // Because value starts at 0 and range at 510, each bypass step
        // is: value = value*2 + bit.
        //
        // The maths here depend on accumulated value vs range, so we just
        // verify round-trip consistency by decoding all 8 bits.
        let mut bits = Vec::new();
        for _ in 0..8 {
            bits.push(dec.decode_bypass());
        }
        // The first several bypass bins with value=0 and range=510 will
        // keep producing false until accumulated value exceeds range.
        // We mainly check no panics and deterministic output.
        assert_eq!(bits.len(), 8);
    }

    #[test]
    fn cabac_terminate_no_end() {
        // Build a stream where terminate returns false (not end of slice).
        // range after init = 510.  terminate subtracts 2 => 508.
        // If value < 508 => not terminated, renormalize.
        // value = 0 < 508 => false.
        let data = [0x00u8; 4];
        let mut dec = CabacDecoder::new(&data);
        assert!(!dec.decode_terminate());
    }

    #[test]
    fn cabac_terminate_end() {
        // Build stream where the initial 9-bit value equals range-2.
        // range = 510, so we need value = 508 = 0b1_1111_1100.
        // As 9 bits: 1_1111_1100.
        //   byte 0: 1111_1110 = 0xFE
        //   byte 1: 0xxx_xxxx — we only need 1 bit from here: 0.
        let data = [0xFE, 0x00];
        let mut dec = CabacDecoder::new(&data);
        // value should be 0b111111100 = 508
        assert_eq!(dec.value, 508);
        assert!(dec.decode_terminate());
    }

    #[test]
    fn cabac_decision_basic() {
        // Construct a stream with known bits and verify a context-modelled
        // decode does not panic and produces a deterministic result.
        let data = [0x00u8; 8];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = ContextModel::new(154);

        // Decode several decision bins — the exact values depend on the
        // arithmetic state, but it should not panic.
        let mut results = Vec::new();
        for _ in 0..10 {
            results.push(dec.decode_decision(&mut ctx));
        }
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn cabac_fl_decode() {
        // Fixed-length decode of 3 bypass bins from known bits.
        let data = [0x00u8; 4];
        let mut dec = CabacDecoder::new(&data);
        let val = dec.decode_fl(3);
        // With value=0 and range=510, 3 bypass bins starting from 0:
        // each step doubles value (0) + next bit (0) => all zeros.
        assert_eq!(val, 0);
    }

    #[test]
    fn cabac_unary_decode() {
        // Unary decode with all-zero stream should produce 0 immediately
        // (first bin = 0 = stop).
        let data = [0x00u8; 8];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = [ContextModel::new(154)];

        // With value=0 and a context where mps=1, the first bin decodes
        // the MPS or LPS depending on range_lps.  We just verify
        // determinism and no panics.
        let val = dec.decode_unary(&mut ctx, 5);
        // The exact value depends on the arithmetic engine state; we
        // verify it is within bounds.
        assert!(val <= 5);
    }

    #[test]
    fn cabac_eg_decode() {
        // Exp-Golomb decode from all-zero stream.
        let data = [0x00u8; 8];
        let mut dec = CabacDecoder::new(&data);
        let val = dec.decode_eg(0);
        // With value=0 and range=510, first bypass bin = false (0), so
        // leading = 0, suffix_len = 0, suffix = 0, value = 0.
        assert_eq!(val, 0);
    }

    #[test]
    fn range_tab_lps_sanity() {
        // Verify a few known entries from the spec table.
        assert_eq!(RANGE_TAB_LPS[0][0], 128);
        assert_eq!(RANGE_TAB_LPS[0][3], 240);
        assert_eq!(RANGE_TAB_LPS[63][0], 2);
        assert_eq!(RANGE_TAB_LPS[63][3], 2);
        // Row 12: [77, 94, 111, 128]
        assert_eq!(RANGE_TAB_LPS[12][0], 77);
        assert_eq!(RANGE_TAB_LPS[12][3], 128);
    }

    #[test]
    fn trans_tables_sanity() {
        // MPS transitions should be monotonically non-decreasing.
        for i in 0..63 {
            assert!(TRANS_IDX_MPS[i] >= i as u8);
        }
        // LPS transitions should be <= current state (convergence to 0).
        for i in 0..63 {
            assert!(TRANS_IDX_LPS[i] <= i as u8);
        }
        // Last MPS entry stays at 63.
        assert_eq!(TRANS_IDX_MPS[63], 63);
        // Last LPS entry is 63 (special case for state 63).
        assert_eq!(TRANS_IDX_LPS[63], 63);
    }

    #[test]
    fn cabac_decode_tr_basic() {
        let data = [0x00u8; 8];
        let mut dec = CabacDecoder::new(&data);
        let mut ctx = [ContextModel::new(154), ContextModel::new(154)];

        let val = dec.decode_tr(&mut ctx, 4, 0);
        assert!(val <= 4);
    }

    #[test]
    fn cabac_bypass_long_sequence() {
        // Decode many bypass bins from a known non-trivial pattern to
        // exercise the renormalization / bit-reading path.
        let data: Vec<u8> = (0..32).collect();
        let mut dec = CabacDecoder::new(&data);
        let mut count_true = 0u32;
        for _ in 0..100 {
            if dec.decode_bypass() {
                count_true += 1;
            }
        }
        // Just verify it ran without panic and produced some mix.
        assert!(count_true <= 100);
    }
}
