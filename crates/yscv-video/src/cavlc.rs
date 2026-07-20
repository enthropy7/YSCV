//! H.264 CAVLC (Context-Adaptive Variable-Length Coding) entropy decoding.
//!
//! CAVLC is the entropy coding method used in H.264 Baseline profile for
//! encoding residual transform coefficients. This module provides a bitstream
//! reader with Exp-Golomb support and a CAVLC block decoder.
//!
//! The coeff_token / total_zeros / run_before VLC tables and the level and nC
//! derivation below are the ITU-T H.264 Tables 9-5, 9-7, 9-9 and 9-10 and the
//! processes of clauses 9.2.1 / 9.2.2 / 9.2.3.

// ---------------------------------------------------------------------------
// BitReader — bit-level access to a byte slice (MSB first)
// ---------------------------------------------------------------------------

/// Reads bits from a byte slice in MSB-first order, with Exp-Golomb support.
pub struct BitReader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) byte_pos: usize,
    pub(crate) bit_pos: u8, // 0..8, bits consumed in current byte
}

impl<'a> BitReader<'a> {
    /// Creates a new `BitReader` over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Creates a `BitReader` positioned mid-stream (continuing after a
    /// header parsed by another reader).
    pub fn new_at(data: &'a [u8], byte_pos: usize, bit_pos: u8) -> Self {
        Self {
            data,
            byte_pos,
            bit_pos,
        }
    }

    /// Skips to the next byte boundary (I_PCM alignment).
    pub fn align_byte(&mut self) {
        if self.bit_pos != 0 {
            self.consume(8 - self.bit_pos);
        }
    }

    /// Whether more RBSP syntax elements follow (clause 7.4.1): true when any
    /// bit after the current position precedes the final RBSP stop bit.
    pub fn more_rbsp_data(&self) -> bool {
        let cur = self.byte_pos * 8 + self.bit_pos as usize;
        for (i, &b) in self.data.iter().enumerate().rev() {
            if b != 0 {
                let last_one = i * 8 + 7 - b.trailing_zeros() as usize;
                return cur < last_one;
            }
        }
        false
    }

    /// Returns the number of unconsumed bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }

    /// Reads `n` bits (1..=32) as a `u32`, MSB first. Returns `None` on exhaustion.
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if n > 32 || self.bits_remaining() < n as usize {
            return None;
        }
        let value = self.peek_padded(n);
        self.consume(n);
        Some(value)
    }

    /// Peek at the next `n` bits without consuming them.
    pub fn peek_bits(&self, n: u8) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if n > 32 || self.bits_remaining() < n as usize {
            return None;
        }
        Some(self.peek_padded(n))
    }

    /// Peek up to `n` bits, left-justified into an `n`-bit window, zero-padded
    /// when fewer than `n` bits remain (VLC lookahead never over-reads a valid
    /// codeword, and trailing zero-padding matches the RBSP stop-bit region).
    /// One unaligned big-endian u64 load covers any 32-bit window.
    fn peek_padded(&self, n: u8) -> u32 {
        debug_assert!((1..=32).contains(&n));
        let tail = &self.data[self.byte_pos.min(self.data.len())..];
        let w = match tail.first_chunk::<8>() {
            Some(chunk) => u64::from_be_bytes(*chunk),
            None => {
                let mut buf = [0u8; 8];
                buf[..tail.len()].copy_from_slice(tail);
                u64::from_be_bytes(buf)
            }
        };
        ((w << self.bit_pos) >> (64 - n as u32)) as u32
    }

    /// Consume (skip) `n` bits.
    pub fn consume(&mut self, n: u8) {
        let total = self.byte_pos * 8 + self.bit_pos as usize + n as usize;
        self.byte_pos = total / 8;
        self.bit_pos = (total % 8) as u8;
    }

    /// Reads an unsigned Exp-Golomb coded integer (ue(v)).
    pub fn read_ue(&mut self) -> Option<u32> {
        // Fast path: codewords up to 31 bits fit the 32-bit peek window, so a
        // single leading-zeros count decodes prefix and suffix together.
        let w = self.peek_padded(32);
        let lz = w.leading_zeros();
        if lz <= 15 {
            let total = 2 * lz + 1;
            if total as usize > self.bits_remaining() {
                return None; // suffix would read past the data
            }
            self.consume(total as u8);
            return Some((w >> (32 - total)) - 1);
        }
        // Slow path: ≥ 33-bit codeword or exhaustion.
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_bits(1)?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None;
            }
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Some((1 << leading_zeros) - 1 + suffix)
    }

    /// Reads a signed Exp-Golomb coded integer (se(v)).
    pub fn read_se(&mut self) -> Option<i32> {
        let code = self.read_ue()?;
        let value = code.div_ceil(2) as i32;
        if code % 2 == 0 {
            Some(-value)
        } else {
            Some(value)
        }
    }
}

// ---------------------------------------------------------------------------
// CAVLC result
// ---------------------------------------------------------------------------

/// Decoded CAVLC residual coefficients for one block.
/// Uses fixed-size arrays (max 16 coefficients) to avoid heap allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CavlcResult {
    /// Number of non-zero coefficients (0..=16).
    pub total_coeffs: usize,
    /// Number of trailing +/-1 coefficients (0..=3).
    pub trailing_ones: usize,
    /// Non-zero coefficient levels in reverse scan order (first `total_coeffs` valid).
    pub levels: [i32; 16],
    /// Total number of zero-valued coefficients before the last non-zero.
    pub total_zeros: usize,
    /// Run of zeros before each coefficient (first `total_coeffs` valid;
    /// run_before ≤ 14 per Table 9-10, so u8 keeps the by-value struct small).
    pub runs: [u8; 16],
}

// ---------------------------------------------------------------------------
// ITU-T H.264 VLC tables (Table 9-5 coeff_token, 9-7/9-9 total_zeros, 9-10
// run_before). Layout ported verbatim from the standard: coeff_token is
// indexed [category][TotalCoeff*4 + TrailingOnes]; a length of 0 marks an
// invalid (TotalCoeff, TrailingOnes) combination.
// ---------------------------------------------------------------------------

// coeff_token, categories: 0 => 0<=nC<2, 1 => 2<=nC<4, 2 => 4<=nC<8, 3 => nC>=8.
#[rustfmt::skip]
const COEFF_TOKEN_LEN: [[u8; 68]; 4] = [
    [
         1, 0, 0, 0,
         6, 2, 0, 0,     8, 6, 3, 0,     9, 8, 7, 5,    10, 9, 8, 6,
        11,10, 9, 7,    13,11,10, 8,    13,13,11, 9,    13,13,13,10,
        14,14,13,11,    14,14,14,13,    15,15,14,14,    15,15,15,14,
        16,15,15,15,    16,16,16,15,    16,16,16,16,    16,16,16,16,
    ],
    [
         2, 0, 0, 0,
         6, 2, 0, 0,     6, 5, 3, 0,     7, 6, 6, 4,     8, 6, 6, 4,
         8, 7, 7, 5,     9, 8, 8, 6,    11, 9, 9, 6,    11,11,11, 7,
        12,11,11, 9,    12,12,12,11,    12,12,12,11,    13,13,13,12,
        13,13,13,13,    13,14,13,13,    14,14,14,13,    14,14,14,14,
    ],
    [
         4, 0, 0, 0,
         6, 4, 0, 0,     6, 5, 4, 0,     6, 5, 5, 4,     7, 5, 5, 4,
         7, 5, 5, 4,     7, 6, 6, 4,     7, 6, 6, 4,     8, 7, 7, 5,
         8, 8, 7, 6,     9, 8, 8, 7,     9, 9, 8, 8,     9, 9, 9, 8,
        10, 9, 9, 9,    10,10,10,10,    10,10,10,10,    10,10,10,10,
    ],
    [
         6, 0, 0, 0,
         6, 6, 0, 0,     6, 6, 6, 0,     6, 6, 6, 6,     6, 6, 6, 6,
         6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
         6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
         6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
    ],
];

#[rustfmt::skip]
const COEFF_TOKEN_BITS: [[u8; 68]; 4] = [
    [
         1, 0, 0, 0,
         5, 1, 0, 0,     7, 4, 1, 0,     7, 6, 5, 3,     7, 6, 5, 3,
         7, 6, 5, 4,    15, 6, 5, 4,    11,14, 5, 4,     8,10,13, 4,
        15,14, 9, 4,    11,10,13,12,    15,14, 9,12,    11,10,13, 8,
        15, 1, 9,12,    11,14,13, 8,     7,10, 9,12,     4, 6, 5, 8,
    ],
    [
         3, 0, 0, 0,
        11, 2, 0, 0,     7, 7, 3, 0,     7,10, 9, 5,     7, 6, 5, 4,
         4, 6, 5, 6,     7, 6, 5, 8,    15, 6, 5, 4,    11,14,13, 4,
        15,10, 9, 4,    11,14,13,12,     8,10, 9, 8,    15,14,13,12,
        11,10, 9,12,     7,11, 6, 8,     9, 8,10, 1,     7, 6, 5, 4,
    ],
    [
        15, 0, 0, 0,
        15,14, 0, 0,    11,15,13, 0,     8,12,14,12,    15,10,11,11,
        11, 8, 9,10,     9,14,13, 9,     8,10, 9, 8,    15,14,13,13,
        11,14,10,12,    15,10,13,12,    11,14, 9,12,     8,10,13, 8,
        13, 7, 9,12,     9,12,11,10,     5, 8, 7, 6,     1, 4, 3, 2,
    ],
    [
         3, 0, 0, 0,
         0, 1, 0, 0,     4, 5, 6, 0,     8, 9,10,11,    12,13,14,15,
        16,17,18,19,    20,21,22,23,    24,25,26,27,    28,29,30,31,
        32,33,34,35,    36,37,38,39,    40,41,42,43,    44,45,46,47,
        48,49,50,51,    52,53,54,55,    56,57,58,59,    60,61,62,63,
    ],
];

// chroma DC coeff_token (ChromaArrayType == 1): indexed [TotalCoeff*4 + T1], TC 0..4.
#[rustfmt::skip]
const CHROMA_DC_COEFF_TOKEN_LEN: [u8; 20] = [
    2, 0, 0, 0,  6, 1, 0, 0,  6, 6, 3, 0,  6, 7, 7, 6,  6, 8, 8, 7,
];
#[rustfmt::skip]
const CHROMA_DC_COEFF_TOKEN_BITS: [u8; 20] = [
    1, 0, 0, 0,  7, 1, 0, 0,  4, 6, 1, 0,  3, 3, 2, 5,  2, 3, 2, 0,
];

// total_zeros for 4x4 blocks (Table 9-7), indexed [TotalCoeff-1][total_zeros].
#[rustfmt::skip]
const TOTAL_ZEROS_LEN: [[u8; 16]; 15] = [
    [1,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9],
    [3,3,3,3,3,4,4,4,4,5,5,6,6,6,6,0],
    [4,3,3,3,4,4,3,3,4,5,5,6,5,6,0,0],
    [5,3,4,4,3,3,3,4,3,4,5,5,5,0,0,0],
    [4,4,4,3,3,3,3,3,4,5,4,5,0,0,0,0],
    [6,5,3,3,3,3,3,3,4,3,6,0,0,0,0,0],
    [6,5,3,3,3,2,3,4,3,6,0,0,0,0,0,0],
    [6,4,5,3,2,2,3,3,6,0,0,0,0,0,0,0],
    [6,6,4,2,2,3,2,5,0,0,0,0,0,0,0,0],
    [5,5,3,2,2,2,4,0,0,0,0,0,0,0,0,0],
    [4,4,3,3,1,3,0,0,0,0,0,0,0,0,0,0],
    [4,4,2,1,3,0,0,0,0,0,0,0,0,0,0,0],
    [3,3,1,2,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
#[rustfmt::skip]
const TOTAL_ZEROS_BITS: [[u8; 16]; 15] = [
    [1,3,2,3,2,3,2,3,2,3,2,3,2,3,2,1],
    [7,6,5,4,3,5,4,3,2,3,2,3,2,1,0,0],
    [5,7,6,5,4,3,4,3,2,3,2,1,1,0,0,0],
    [3,7,5,4,6,5,4,3,3,2,2,1,0,0,0,0],
    [5,4,3,7,6,5,4,3,2,1,1,0,0,0,0,0],
    [1,1,7,6,5,4,3,2,1,1,0,0,0,0,0,0],
    [1,1,5,4,3,3,2,1,1,0,0,0,0,0,0,0],
    [1,1,1,3,3,2,2,1,0,0,0,0,0,0,0,0],
    [1,0,1,3,2,1,1,1,0,0,0,0,0,0,0,0],
    [1,0,1,3,2,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,2,1,3,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// total_zeros for chroma DC 2x2 (Table 9-9(a)), indexed [TotalCoeff-1][total_zeros].
#[rustfmt::skip]
const CHROMA_DC_TOTAL_ZEROS_LEN: [[u8; 4]; 3] = [
    [1,2,3,3],
    [1,2,2,0],
    [1,1,0,0],
];
#[rustfmt::skip]
const CHROMA_DC_TOTAL_ZEROS_BITS: [[u8; 4]; 3] = [
    [1,1,1,0],
    [1,1,0,0],
    [1,0,0,0],
];

// run_before (Table 9-10), indexed [min(zerosLeft,7)-1][run_before].
#[rustfmt::skip]
const RUN_LEN: [[u8; 16]; 7] = [
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,3,3,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,3,3,3,3,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0],
    [3,3,3,3,3,3,3,4,5,6,7,8,9,10,11,0],
];
#[rustfmt::skip]
const RUN_BITS: [[u8; 16]; 7] = [
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,3,2,1,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,1,3,2,5,4,0,0,0,0,0,0,0,0,0],
    [7,6,5,4,3,2,1,1,1,1,1,1,1,1,1,0],
];

// ---------------------------------------------------------------------------
// VLC lookup helpers — direct 9-bit table lookup built at compile time from
// the (len, bits) pairs, with a linear-scan fallback for longer codewords.
// ---------------------------------------------------------------------------

/// Window width of the direct-lookup tables. Every total_zeros / run_before /
/// chroma-DC codeword fits; only the rare long coeff_token (≤16 bits) and
/// run_before zerosLeft>6 (≤11 bits) codes take the fallback scan.
const VLC_LUT_BITS: u8 = 9;

/// Direct VLC lookup: indexed by the next 9 bits, each entry holds
/// `(table_index << 5) | code_len`, or 0 when no codeword of length ≤ 9
/// matches that prefix (longer codeword or invalid bitstream).
type VlcLut = [u16; 1 << VLC_LUT_BITS];

/// Expands one (len, bits) table into a [`VlcLut`] (compile-time).
const fn build_vlc_lut<const N: usize>(lens: &[u8; N], bits: &[u8; N]) -> VlcLut {
    let mut lut = [0u16; 1 << VLC_LUT_BITS];
    let mut i = 0;
    while i < N {
        let len = lens[i];
        if len > 0 && len <= VLC_LUT_BITS {
            // Every window starting with this codeword maps to entry i.
            let lo = (bits[i] as usize) << (VLC_LUT_BITS - len);
            let hi = ((bits[i] as usize) + 1) << (VLC_LUT_BITS - len);
            let mut w = lo;
            while w < hi {
                lut[w] = ((i as u16) << 5) | len as u16;
                w += 1;
            }
        }
        i += 1;
    }
    lut
}

/// Builds one [`VlcLut`] per row of a two-dimensional VLC table (compile-time).
const fn build_vlc_lut_rows<const C: usize, const R: usize>(
    lens: &[[u8; C]; R],
    bits: &[[u8; C]; R],
) -> [VlcLut; R] {
    let mut out = [[0u16; 1 << VLC_LUT_BITS]; R];
    let mut r = 0;
    while r < R {
        out[r] = build_vlc_lut(&lens[r], &bits[r]);
        r += 1;
    }
    out
}

static COEFF_TOKEN_LUT: [VlcLut; 4] = build_vlc_lut_rows(&COEFF_TOKEN_LEN, &COEFF_TOKEN_BITS);
static CHROMA_DC_COEFF_TOKEN_LUT: VlcLut =
    build_vlc_lut(&CHROMA_DC_COEFF_TOKEN_LEN, &CHROMA_DC_COEFF_TOKEN_BITS);
static TOTAL_ZEROS_LUT: [VlcLut; 15] = build_vlc_lut_rows(&TOTAL_ZEROS_LEN, &TOTAL_ZEROS_BITS);
static CHROMA_DC_TOTAL_ZEROS_LUT: [VlcLut; 3] =
    build_vlc_lut_rows(&CHROMA_DC_TOTAL_ZEROS_LEN, &CHROMA_DC_TOTAL_ZEROS_BITS);
static RUN_LUT: [VlcLut; 7] = build_vlc_lut_rows(&RUN_LEN, &RUN_BITS);

/// Match a prefix-free VLC via its direct-lookup table, falling back to a
/// longest-match scan over the parallel `len`/`bits` arrays for codewords
/// longer than the lookup window. Consumes the matched bits.
fn match_vlc(reader: &mut BitReader, lens: &[u8], bits: &[u8], lut: &VlcLut) -> Option<usize> {
    let entry = lut[reader.peek_padded(VLC_LUT_BITS) as usize];
    if entry != 0 {
        reader.consume((entry & 31) as u8);
        return Some((entry >> 5) as usize);
    }
    // Rare: codewords longer than the lookup window (or invalid bits).
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return None;
    }
    let window = reader.peek_padded(max_len);
    for (i, (&len, &pat)) in lens.iter().zip(bits.iter()).enumerate() {
        if len == 0 {
            continue;
        }
        if (window >> (max_len - len)) == pat as u32 {
            reader.consume(len);
            return Some(i);
        }
    }
    None
}

/// Reads coeff_token, returning (total_coeffs, trailing_ones).
/// `nc == -1` selects the chroma-DC (ChromaArrayType 1) table.
fn read_coeff_token(reader: &mut BitReader, nc: i32) -> Option<(usize, usize)> {
    let idx = if nc < 0 {
        match_vlc(
            reader,
            &CHROMA_DC_COEFF_TOKEN_LEN,
            &CHROMA_DC_COEFF_TOKEN_BITS,
            &CHROMA_DC_COEFF_TOKEN_LUT,
        )?
    } else {
        let cat = match nc {
            0..=1 => 0,
            2..=3 => 1,
            4..=7 => 2,
            _ => 3,
        };
        match_vlc(
            reader,
            &COEFF_TOKEN_LEN[cat],
            &COEFF_TOKEN_BITS[cat],
            &COEFF_TOKEN_LUT[cat],
        )?
    };
    Some((idx >> 2, idx & 3))
}

/// Reads total_zeros for a block with `total_coeff` non-zero coefficients.
/// `max_coeff == 4` uses the chroma-DC table; otherwise the 4x4 table.
fn read_total_zeros(reader: &mut BitReader, total_coeff: usize, max_coeff: usize) -> Option<usize> {
    if max_coeff == 4 {
        let row = total_coeff - 1;
        match_vlc(
            reader,
            &CHROMA_DC_TOTAL_ZEROS_LEN[row],
            &CHROMA_DC_TOTAL_ZEROS_BITS[row],
            &CHROMA_DC_TOTAL_ZEROS_LUT[row],
        )
    } else {
        let row = total_coeff - 1;
        match_vlc(
            reader,
            &TOTAL_ZEROS_LEN[row],
            &TOTAL_ZEROS_BITS[row],
            &TOTAL_ZEROS_LUT[row],
        )
    }
}

/// Reads run_before given the number of zeros still to be distributed.
fn read_run_before(reader: &mut BitReader, zeros_left: usize) -> Option<usize> {
    if zeros_left == 0 {
        return Some(0);
    }
    let row = zeros_left.min(7) - 1;
    match_vlc(reader, &RUN_LEN[row], &RUN_BITS[row], &RUN_LUT[row])
}

/// Reads a level_prefix: the number of leading zero bits before the terminating 1.
fn read_level_prefix(reader: &mut BitReader) -> Option<u32> {
    // The terminating 1 must be a real data bit (the peek pads with zeros);
    // an all-zero window means exhaustion or prefix > 31 — None either way.
    let w = reader.peek_padded(32);
    let lz = w.leading_zeros();
    if lz >= 32 || lz as usize >= reader.bits_remaining() {
        return None;
    }
    reader.consume(lz as u8 + 1);
    Some(lz)
}

// ---------------------------------------------------------------------------
// Main CAVLC block decoder
// ---------------------------------------------------------------------------

/// Decodes one CAVLC-coded 4x4 luma/AC residual block (16 coefficients).
///
/// `nc` is the predicted number of non-zero coefficients derived from
/// neighbouring blocks (used to select the coeff_token VLC table).
pub fn decode_cavlc_block(reader: &mut BitReader, nc: i32) -> Option<CavlcResult> {
    decode_cavlc_block_max(reader, nc, 16)
}

/// Decodes one CAVLC-coded residual block with `max_coeff` coefficients
/// (16 for luma 4x4, 15 for AC blocks, 4 for chroma DC — pass `nc == -1` for
/// the chroma-DC coeff_token table). Implements ITU-T H.264 clause 9.2.
pub fn decode_cavlc_block_max(
    reader: &mut BitReader,
    nc: i32,
    max_coeff: usize,
) -> Option<CavlcResult> {
    // (a) coeff_token -> (total_coeff, trailing_ones)
    let (total_coeffs, trailing_ones) = read_coeff_token(reader, nc)?;
    if total_coeffs > max_coeff {
        return None;
    }

    if total_coeffs == 0 {
        return Some(CavlcResult {
            total_coeffs: 0,
            trailing_ones: 0,
            levels: [0; 16],
            total_zeros: 0,
            runs: [0; 16],
        });
    }

    let mut levels = [0i32; 16];
    let mut level_count = 0usize;

    // (b) sign of each trailing one: read all of them in one go.
    if trailing_ones > 0 {
        let signs = reader.read_bits(trailing_ones as u8)?;
        for k in 0..trailing_ones {
            let bit = (signs >> (trailing_ones - 1 - k)) & 1;
            levels[level_count] = if bit == 0 { 1 } else { -1 };
            level_count += 1;
        }
    }

    // (c) remaining levels (clause 9.2.2.1).
    let mut suffix_length: u32 = if total_coeffs > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };

    for i in trailing_ones..total_coeffs {
        // Fast path: for prefixes ≤ 13 the suffix size equals suffix_length
        // (≤ 6), so prefix + stop bit + suffix (≤ 20 bits) all sit in one
        // 32-bit peek and consume together.
        let w = reader.peek_padded(32);
        let lz = w.leading_zeros();
        let total = lz + 1 + suffix_length;
        let (level_prefix, level_suffix);
        if lz <= 13 && total as usize <= reader.bits_remaining() {
            level_prefix = lz;
            level_suffix = (w >> (32 - total)) & ((1u32 << suffix_length) - 1);
            reader.consume(total as u8);
        } else {
            level_prefix = read_level_prefix(reader)?;
            let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
                4
            } else if level_prefix >= 15 {
                level_prefix - 3
            } else {
                suffix_length
            };
            level_suffix = if level_suffix_size > 0 {
                reader.read_bits(level_suffix_size as u8)?
            } else {
                0
            };
        }

        let mut level_code = (level_prefix.min(15) << suffix_length) as i32 + level_suffix as i32;
        if level_prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if level_prefix >= 16 {
            level_code += ((1u32 << (level_prefix - 3)) - 4096) as i32;
        }
        // The first coefficient after the trailing ones adds 2 when fewer than
        // three trailing ones were present.
        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }

        let level = if level_code % 2 == 0 {
            (level_code + 2) >> 1
        } else {
            (-level_code - 1) >> 1
        };

        levels[level_count] = level;
        level_count += 1;

        if suffix_length == 0 {
            suffix_length = 1;
        }
        if level.unsigned_abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // (d) total_zeros.
    let total_zeros = if total_coeffs < max_coeff {
        read_total_zeros(reader, total_coeffs, max_coeff)?
    } else {
        0
    };

    // (e) run_before for each coefficient except the last.
    let mut runs = [0u8; 16];
    let mut zeros_left = total_zeros;
    for run in runs.iter_mut().take(total_coeffs - 1) {
        if zeros_left == 0 {
            break;
        }
        let r = read_run_before(reader, zeros_left)?;
        *run = r as u8;
        zeros_left = zeros_left.saturating_sub(r);
    }
    runs[total_coeffs - 1] = zeros_left as u8;

    Some(CavlcResult {
        total_coeffs,
        trailing_ones,
        levels,
        total_zeros,
        runs,
    })
}

// ---------------------------------------------------------------------------
// Coefficient expansion
// ---------------------------------------------------------------------------

/// Expands a `CavlcResult` into a full coefficient array of `block_size`
/// elements, inserting zero runs at the correct positions.
///
/// Coefficients in the `CavlcResult` are stored in reverse scan order
/// (highest frequency first). This function places them into forward scan
/// order (DC first) and inserts the decoded zero runs.
pub fn expand_cavlc_to_coefficients(result: &CavlcResult, block_size: usize) -> Vec<i32> {
    let mut coeffs = vec![0i32; block_size];
    expand_cavlc_to_coefficients_into(result, &mut coeffs);
    coeffs
}

/// Zero-allocation version: writes coefficients into a pre-allocated slice.
/// Slice must be zeroed before calling.
pub fn expand_cavlc_to_coefficients_into(result: &CavlcResult, coeffs: &mut [i32]) {
    let n = result.total_coeffs;
    if n == 0 {
        return;
    }

    // Clause 9.2.4: coefficients are decoded highest-frequency first, so we walk
    // the level/run arrays in reverse (from the last-decoded DC-side coefficient
    // upward), advancing the scan position by `run_before + 1` each step. This
    // anchors the DC at position 0 regardless of how many coefficients are
    // present — filling from the top only happens to be correct for a full block.
    let mut pos: i32 = -1;
    for i in (0..n).rev() {
        let run = result.runs.get(i).copied().unwrap_or(0);
        pos += run as i32 + 1;
        match (usize::try_from(pos), result.levels.get(i)) {
            (Ok(p), Some(&level)) if p < coeffs.len() => coeffs[p] = level,
            _ => break,
        }
    }
}

#[cfg(test)]
mod cavlc_tests {
    use super::*;

    // Build a BitReader from an MSB-first bit string like "000101".
    fn br(bits: &str) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, c) in bits.chars().enumerate() {
            if c == '1' {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    fn ct(nc: i32, bits: &str) -> (usize, usize) {
        let data = br(bits);
        let mut r = BitReader::new(&data);
        read_coeff_token(&mut r, nc).expect("coeff_token match")
    }

    #[test]
    fn coeff_token_spec_codewords() {
        // Table 9-5, 0 <= nC < 2.
        assert_eq!(ct(0, "1"), (0, 0));
        assert_eq!(ct(0, "01"), (1, 1));
        assert_eq!(ct(0, "001"), (2, 2));
        assert_eq!(ct(0, "000101"), (1, 0));
        assert_eq!(ct(0, "0000101"), (3, 2));
        assert_eq!(ct(0, "00011"), (3, 3));
        // 2 <= nC < 4.
        assert_eq!(ct(2, "11"), (0, 0));
        assert_eq!(ct(2, "10"), (1, 1));
        assert_eq!(ct(2, "011"), (2, 2));
        // 4 <= nC < 8.
        assert_eq!(ct(4, "1111"), (0, 0));
        assert_eq!(ct(4, "1110"), (1, 1));
        // nC >= 8 (6-bit FLC): value = (tc-1)*4 + t1, and (0,0) == 000011.
        assert_eq!(ct(8, "000011"), (0, 0));
        assert_eq!(ct(8, "000000"), (1, 0));
        assert_eq!(ct(8, "000101"), (2, 1));
        // chroma DC (nc == -1): (0,0) is "01", (1,1) is "1".
        assert_eq!(ct(-1, "01"), (0, 0));
        assert_eq!(ct(-1, "1"), (1, 1));
    }

    #[test]
    fn total_zeros_and_run_spec() {
        // total_zeros Table 9-7, tc=1: code "1" -> 0 zeros; "011" -> 1.
        {
            let d = br("1");
            let mut r = BitReader::new(&d);
            assert_eq!(read_total_zeros(&mut r, 1, 16).unwrap(), 0);
        }
        {
            let d = br("011");
            let mut r = BitReader::new(&d);
            assert_eq!(read_total_zeros(&mut r, 1, 16).unwrap(), 1);
        }
        // run_before Table 9-10, zerosLeft=1: "1"->0, "0"->1.
        {
            let d = br("1");
            let mut r = BitReader::new(&d);
            assert_eq!(read_run_before(&mut r, 1).unwrap(), 0);
        }
        {
            let d = br("0");
            let mut r = BitReader::new(&d);
            assert_eq!(read_run_before(&mut r, 1).unwrap(), 1);
        }
    }
}
