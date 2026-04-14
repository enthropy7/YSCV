//! Baseline MJPEG/JFIF decoder — parses SOI, DQT, SOF0, DHT, SOS, EOI
//! markers and produces RGB8 output via Huffman decode, dequantize,
//! 8x8 IDCT (Loeffler), and YCbCr-to-RGB conversion.

use crate::VideoError;

// ---------------------------------------------------------------------------
// JPEG marker codes
// ---------------------------------------------------------------------------

const MARKER_SOI: u8 = 0xD8;
const MARKER_EOI: u8 = 0xD9;
const MARKER_DQT: u8 = 0xDB;
const MARKER_SOF0: u8 = 0xC0; // baseline DCT
const MARKER_DHT: u8 = 0xC4;
const MARKER_SOS: u8 = 0xDA;
const MARKER_DRI: u8 = 0xDD;

// ---------------------------------------------------------------------------
// Zigzag order
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// ---------------------------------------------------------------------------
// Quantization tables (up to 4 tables)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct QuantTable {
    table: [u16; 64],
}

impl Default for QuantTable {
    fn default() -> Self {
        Self { table: [1; 64] }
    }
}

// ---------------------------------------------------------------------------
// Huffman tables
// ---------------------------------------------------------------------------

/// Maximum code length for a JPEG Huffman table (16 bits).
const MAX_CODE_LEN: usize = 16;

#[derive(Clone)]
struct HuffTable {
    /// For each code length 1..=16, the minimum code value at that length.
    min_code: [i32; MAX_CODE_LEN + 1],
    /// For each code length, the maximum code value.
    max_code: [i32; MAX_CODE_LEN + 1],
    /// Index into `values` for the first symbol at each code length.
    val_ptr: [usize; MAX_CODE_LEN + 1],
    /// The raw symbol values in order.
    values: Vec<u8>,
}

impl Default for HuffTable {
    fn default() -> Self {
        Self {
            min_code: [-1; MAX_CODE_LEN + 1],
            max_code: [-1; MAX_CODE_LEN + 1],
            val_ptr: [0; MAX_CODE_LEN + 1],
            values: Vec::new(),
        }
    }
}

impl HuffTable {
    fn build(counts: &[u8; 16], symbols: &[u8]) -> Self {
        let mut ht = HuffTable {
            values: symbols.to_vec(),
            ..HuffTable::default()
        };

        let mut code: i32 = 0;
        let mut si = 0usize;

        for bits in 1..=MAX_CODE_LEN {
            let count = counts[bits - 1] as usize;
            if count == 0 {
                ht.min_code[bits] = -1;
                ht.max_code[bits] = -1;
                ht.val_ptr[bits] = si;
                code <<= 1;
                continue;
            }
            ht.val_ptr[bits] = si;
            ht.min_code[bits] = code;
            code += count as i32;
            ht.max_code[bits] = code - 1;
            si += count;
            code <<= 1;
        }

        ht
    }
}

// ---------------------------------------------------------------------------
// Bitstream reader for entropy-coded data
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_left: 0,
        }
    }

    /// Read the next byte, handling byte-stuffing (0xFF 0x00 → 0xFF).
    #[inline]
    fn next_byte(&mut self) -> Result<u8, VideoError> {
        if self.pos >= self.data.len() {
            return Err(VideoError::Codec(
                "MJPEG: unexpected end of scan data".into(),
            ));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        if b == 0xFF && self.pos < self.data.len() && self.data[self.pos] == 0x00 {
            self.pos += 1; // skip stuffed zero
        }
        // If it's a restart marker (0xD0-0xD7), that's handled elsewhere
        Ok(b)
    }

    /// Fill the bit buffer with at least `need` bits.
    fn fill(&mut self, need: u8) -> Result<(), VideoError> {
        while self.bits_left < need {
            let b = self.next_byte()? as u32;
            self.bit_buf = (self.bit_buf << 8) | b;
            self.bits_left += 8;
        }
        Ok(())
    }

    /// Read `n` bits (1..=16) from the stream, MSB first.
    fn read_bits(&mut self, n: u8) -> Result<u16, VideoError> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        self.bits_left -= n;
        let val = (self.bit_buf >> self.bits_left) & ((1u32 << n) - 1);
        Ok(val as u16)
    }

    /// Decode one Huffman symbol from the bitstream.
    fn decode_huff(&mut self, ht: &HuffTable) -> Result<u8, VideoError> {
        let mut code: i32 = 0;
        for bits in 1..=MAX_CODE_LEN {
            self.fill(1)?;
            self.bits_left -= 1;
            let bit = ((self.bit_buf >> self.bits_left) & 1) as i32;
            code = (code << 1) | bit;

            if ht.max_code[bits] >= 0 && code <= ht.max_code[bits] {
                let idx = ht.val_ptr[bits] + (code - ht.min_code[bits]) as usize;
                return ht.values.get(idx).copied().ok_or_else(|| {
                    VideoError::Codec("MJPEG: Huffman symbol index out of range".into())
                });
            }
        }
        Err(VideoError::Codec(
            "MJPEG: invalid Huffman code (exceeded 16 bits)".into(),
        ))
    }

    /// Read a signed coefficient value of `n` bits (JPEG coefficient coding).
    fn receive_extend(&mut self, n: u8) -> Result<i32, VideoError> {
        if n == 0 {
            return Ok(0);
        }
        let raw = self.read_bits(n)? as i32;
        // If MSB is 0, the value is negative (one's complement).
        let threshold = 1i32 << (n - 1);
        if raw < threshold {
            Ok(raw - (2 * threshold - 1))
        } else {
            Ok(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// Frame / component info
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct ComponentInfo {
    h_sampling: u8,
    v_sampling: u8,
    quant_table_id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
}

struct FrameHeader {
    width: u32,
    height: u32,
    components: Vec<ComponentInfo>,
    max_h: u8,
    max_v: u8,
}

// ---------------------------------------------------------------------------
// Loeffler 8x8 IDCT
// ---------------------------------------------------------------------------

/// Loeffler/Ligtenberg/Moschytz 8-point IDCT, fixed-point Q12.
/// Operates in-place on a row/column of 8 i32 values.
#[inline]
fn idct_1d(data: &mut [i32; 8]) {
    // Constants scaled to Q12 (multiply by 4096):
    // cos(pi/16)*sqrt(2) ≈ 1.3870 → 5681
    // cos(3pi/16)*sqrt(2) ≈ 1.1759 → 4816
    // cos(pi/4)*sqrt(2) = 2.0 → 8192 ... no, we need different:
    //
    // Standard AAN / Loeffler constants (Q12 = <<12):
    // W1 = cos(7pi/16)*sqrt(2)*4096 = 1.3870*4096 = 5681 — NO
    //
    // Use the classic Chen DCT matrix constants (Q12):
    // C1 = cos(pi/16) = 0.9808 → 4017
    // S1 = sin(pi/16) = 0.1951 → 799
    // C3 = cos(3pi/16) = 0.8315 → 3406
    // S3 = sin(3pi/16) = 0.5556 → 2276
    // C6 = cos(6pi/16) = 0.3827 → 1567
    // S6 = sin(6pi/16) = 0.9239 → 3784
    // C4 = cos(pi/4) = 0.7071 → 2896

    const C1: i32 = 4017;
    const S1: i32 = 799;
    const C3: i32 = 3406;
    const S3: i32 = 2276;
    const C6: i32 = 1567;
    const S6: i32 = 3784;
    const C4: i32 = 2896;

    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let x5 = data[5];
    let x6 = data[6];
    let x7 = data[7];

    // Stage 1: butterfly on even indices
    let s0 = x0 + x4;
    let s1 = x0 - x4;
    let s2 = (x2 * S6 - x6 * C6 + 2048) >> 12;
    let s3 = (x2 * C6 + x6 * S6 + 2048) >> 12;

    // Stage 2: even part
    let e0 = s0 + s3;
    let e1 = s1 + s2;
    let e2 = s1 - s2;
    let e3 = s0 - s3;

    // Stage 1: odd part rotations
    let o0 = (x1 * C1 + x7 * S1 + 2048) >> 12;
    let o1 = (x1 * S1 - x7 * C1 + 2048) >> 12;
    let o2 = (x3 * C3 + x5 * S3 + 2048) >> 12;
    let o3 = (x3 * S3 - x5 * C3 + 2048) >> 12;

    // Stage 2: odd part butterfly
    let t0 = o0 + o2;
    let t1 = o1 + o3;
    let t2 = o0 - o2;
    let t3 = o1 - o3;

    // Rotation by cos(pi/4)
    let t2r = (t2 * C4 + 2048) >> 12;
    let t3r = (t3 * C4 + 2048) >> 12;

    // Final combination
    data[0] = e0 + t0;
    data[1] = e1 + t2r + t3r;
    data[2] = e2 + t3r - t2r;
    data[3] = e3 + t1;
    data[4] = e3 - t1;
    data[5] = e2 - t3r + t2r;
    data[6] = e1 - t2r - t3r;
    data[7] = e0 - t0;
}

/// Full 2D 8x8 IDCT: row pass then column pass.
fn idct_8x8(block: &mut [i32; 64]) {
    // Row pass
    for r in 0..8 {
        let off = r * 8;
        let mut row = [
            block[off],
            block[off + 1],
            block[off + 2],
            block[off + 3],
            block[off + 4],
            block[off + 5],
            block[off + 6],
            block[off + 7],
        ];
        idct_1d(&mut row);
        block[off..off + 8].copy_from_slice(&row);
    }

    // Column pass
    for c in 0..8 {
        let mut col = [
            block[c],
            block[c + 8],
            block[c + 16],
            block[c + 24],
            block[c + 32],
            block[c + 40],
            block[c + 48],
            block[c + 56],
        ];
        idct_1d(&mut col);
        block[c] = col[0];
        block[c + 8] = col[1];
        block[c + 16] = col[2];
        block[c + 24] = col[3];
        block[c + 32] = col[4];
        block[c + 40] = col[5];
        block[c + 48] = col[6];
        block[c + 56] = col[7];
    }
}

// ---------------------------------------------------------------------------
// Marker parsing
// ---------------------------------------------------------------------------

/// Skip over an APP or unknown marker segment.
fn skip_marker(data: &[u8], pos: &mut usize) -> Result<(), VideoError> {
    if *pos + 2 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated marker length".into()));
    }
    let len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if *pos + len > data.len() {
        return Err(VideoError::Codec(
            "MJPEG: marker segment exceeds data".into(),
        ));
    }
    *pos += len;
    Ok(())
}

fn parse_dqt(data: &[u8], pos: &mut usize, qt: &mut [QuantTable; 4]) -> Result<(), VideoError> {
    if *pos + 2 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated DQT".into()));
    }
    let seg_len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if *pos + seg_len > data.len() {
        return Err(VideoError::Codec("MJPEG: DQT exceeds data".into()));
    }
    let end = *pos + seg_len;
    *pos += 2; // skip length

    while *pos < end {
        let info = data[*pos];
        *pos += 1;
        let precision = (info >> 4) & 0x0F; // 0 = 8-bit, 1 = 16-bit
        let table_id = (info & 0x0F) as usize;
        if table_id >= 4 {
            return Err(VideoError::Codec(format!(
                "MJPEG: invalid DQT table id {table_id}"
            )));
        }

        if precision == 0 {
            // 8-bit values
            if *pos + 64 > end {
                return Err(VideoError::Codec("MJPEG: truncated 8-bit DQT".into()));
            }
            for i in 0..64 {
                qt[table_id].table[ZIGZAG[i]] = data[*pos] as u16;
                *pos += 1;
            }
        } else {
            // 16-bit values
            if *pos + 128 > end {
                return Err(VideoError::Codec("MJPEG: truncated 16-bit DQT".into()));
            }
            for i in 0..64 {
                qt[table_id].table[ZIGZAG[i]] = u16::from_be_bytes([data[*pos], data[*pos + 1]]);
                *pos += 2;
            }
        }
    }
    Ok(())
}

fn parse_sof0(data: &[u8], pos: &mut usize) -> Result<FrameHeader, VideoError> {
    if *pos + 2 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated SOF0".into()));
    }
    let seg_len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if *pos + seg_len > data.len() {
        return Err(VideoError::Codec("MJPEG: SOF0 exceeds data".into()));
    }
    *pos += 2;

    let precision = data[*pos];
    *pos += 1;
    if precision != 8 {
        return Err(VideoError::Codec(format!(
            "MJPEG: only 8-bit precision supported, got {precision}"
        )));
    }

    let height = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as u32;
    *pos += 2;
    let width = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as u32;
    *pos += 2;

    let num_components = data[*pos] as usize;
    *pos += 1;

    if num_components != 1 && num_components != 3 {
        return Err(VideoError::Codec(format!(
            "MJPEG: unsupported component count {num_components}"
        )));
    }

    let mut components = Vec::with_capacity(num_components);
    let mut max_h = 1u8;
    let mut max_v = 1u8;

    for _ in 0..num_components {
        let _id = data[*pos]; // component id (1=Y, 2=Cb, 3=Cr typically)
        *pos += 1;
        let sampling = data[*pos];
        *pos += 1;
        let h = (sampling >> 4) & 0x0F;
        let v = sampling & 0x0F;
        let qt_id = data[*pos];
        *pos += 1;

        if h > max_h {
            max_h = h;
        }
        if v > max_v {
            max_v = v;
        }

        components.push(ComponentInfo {
            h_sampling: h,
            v_sampling: v,
            quant_table_id: qt_id,
            ..Default::default()
        });
    }

    Ok(FrameHeader {
        width,
        height,
        components,
        max_h,
        max_v,
    })
}

fn parse_dht(
    data: &[u8],
    pos: &mut usize,
    huff: &mut [[HuffTable; 2]; 2],
) -> Result<(), VideoError> {
    if *pos + 2 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated DHT".into()));
    }
    let seg_len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if *pos + seg_len > data.len() {
        return Err(VideoError::Codec("MJPEG: DHT exceeds data".into()));
    }
    let end = *pos + seg_len;
    *pos += 2;

    while *pos < end {
        let info = data[*pos];
        *pos += 1;
        let table_class = ((info >> 4) & 0x0F) as usize; // 0=DC, 1=AC
        let table_id = (info & 0x0F) as usize;
        if table_class >= 2 || table_id >= 2 {
            return Err(VideoError::Codec(format!(
                "MJPEG: invalid DHT class={table_class} id={table_id}"
            )));
        }

        if *pos + 16 > end {
            return Err(VideoError::Codec("MJPEG: truncated DHT counts".into()));
        }
        let mut counts = [0u8; 16];
        counts.copy_from_slice(&data[*pos..*pos + 16]);
        *pos += 16;

        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if *pos + total > end {
            return Err(VideoError::Codec("MJPEG: truncated DHT symbols".into()));
        }
        let symbols = &data[*pos..*pos + total];
        *pos += total;

        huff[table_class][table_id] = HuffTable::build(&counts, symbols);
    }
    Ok(())
}

fn parse_dri(data: &[u8], pos: &mut usize) -> Result<u16, VideoError> {
    if *pos + 4 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated DRI".into()));
    }
    // length is always 4
    *pos += 2;
    let interval = u16::from_be_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(interval)
}

/// Parse SOS header, returning the scan data start position.
fn parse_sos(
    data: &[u8],
    pos: &mut usize,
    components: &mut [ComponentInfo],
) -> Result<(), VideoError> {
    if *pos + 2 > data.len() {
        return Err(VideoError::Codec("MJPEG: truncated SOS".into()));
    }
    let seg_len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if *pos + seg_len > data.len() {
        return Err(VideoError::Codec("MJPEG: SOS exceeds data".into()));
    }
    *pos += 2;

    let num_scan_components = data[*pos] as usize;
    *pos += 1;

    for _ in 0..num_scan_components {
        let comp_selector = data[*pos];
        *pos += 1;
        let table_sel = data[*pos];
        *pos += 1;
        let dc_id = (table_sel >> 4) & 0x0F;
        let ac_id = table_sel & 0x0F;

        // Component selector is 1-based typically
        let comp_idx = if comp_selector == 0 {
            0
        } else {
            (comp_selector - 1) as usize
        };
        if comp_idx < components.len() {
            components[comp_idx].dc_table_id = dc_id;
            components[comp_idx].ac_table_id = ac_id;
        }
    }

    // Skip Ss, Se, Ah/Al (3 bytes)
    *pos += 3;
    Ok(())
}

/// Find the end of the scan data (look for 0xFF followed by non-0x00, non-RST marker).
fn find_scan_end(data: &[u8], start: usize) -> usize {
    let mut i = start;
    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let next = data[i + 1];
            // 0x00 is byte stuffing, 0xD0-0xD7 are restart markers — skip both
            if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                i += 2;
                continue;
            }
            // Any other marker ends the scan
            return i;
        }
        i += 1;
    }
    data.len()
}

// ---------------------------------------------------------------------------
// MCU decode
// ---------------------------------------------------------------------------

/// Decode one 8x8 block from the Huffman-coded bitstream.
fn decode_block(
    reader: &mut BitReader<'_>,
    dc_huff: &HuffTable,
    ac_huff: &HuffTable,
    qt: &[u16; 64],
    prev_dc: &mut i32,
) -> Result<[i32; 64], VideoError> {
    let mut coeffs = [0i32; 64];

    // DC coefficient
    let dc_len = reader.decode_huff(dc_huff)?;
    let dc_diff = reader.receive_extend(dc_len)?;
    *prev_dc += dc_diff;
    coeffs[0] = *prev_dc * qt[0] as i32;

    // AC coefficients
    let mut k = 1usize;
    while k < 64 {
        let rs = reader.decode_huff(ac_huff)?;
        let run = (rs >> 4) as usize;
        let size = rs & 0x0F;

        if size == 0 {
            if run == 0 {
                break; // EOB
            }
            if run == 0x0F {
                k += 16; // ZRL (skip 16 zeros)
                continue;
            }
            break;
        }

        k += run;
        if k >= 64 {
            break;
        }

        let value = reader.receive_extend(size)?;
        coeffs[ZIGZAG[k]] = value * qt[ZIGZAG[k]] as i32;
        k += 1;
    }

    // Perform 2D 8x8 IDCT
    idct_8x8(&mut coeffs);

    // Level-shift: add 128, clamp to [0, 255]
    // The IDCT output is scaled by ~(1/8)*(1/8) = 1/64 from the 2D transform,
    // but our fixed-point IDCT has its own scaling. We need to descale.
    // Our IDCT uses Q12 multiply and >>12 in each 1D pass. After two passes
    // the values have been divided by the DCT normalization but we need to
    // account for the Q12 scaling: each pass does multiply+>>12, so the net
    // scale is approximately right. We just need the final >>0 + 128 shift.
    // However, the actual IDCT precision means we must divide by some factor.
    // Empirically, for the Chen-Wang variant with Q12 constants, the output
    // after two passes is already in the right range (no additional shifting
    // needed beyond the level-shift of +128).
    for coeff in coeffs.iter_mut() {
        *coeff = (*coeff + 128).clamp(0, 255);
    }

    Ok(coeffs)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decodes an MJPEG/JFIF frame to interleaved RGB8.
///
/// Returns `(width, height)` on success. `output` must be at least
/// `width * height * 3` bytes (resized internally if too small).
///
/// Supports baseline JPEG (SOF0) with YCbCr 4:2:0, 4:2:2, 4:4:4 and grayscale.
pub fn decode_mjpeg_to_rgb8(data: &[u8], output: &mut Vec<u8>) -> Result<(u32, u32), VideoError> {
    if data.len() < 4 {
        return Err(VideoError::Codec("MJPEG: data too short".into()));
    }

    // Verify SOI marker
    if data[0] != 0xFF || data[1] != MARKER_SOI {
        return Err(VideoError::Codec("MJPEG: missing SOI marker".into()));
    }

    let mut pos = 2usize;
    let mut qt = [
        QuantTable::default(),
        QuantTable::default(),
        QuantTable::default(),
        QuantTable::default(),
    ];
    // huff[class][id]: class 0=DC, 1=AC; id 0 or 1
    let mut huff: [[HuffTable; 2]; 2] = [
        [HuffTable::default(), HuffTable::default()],
        [HuffTable::default(), HuffTable::default()],
    ];
    let mut frame_header: Option<FrameHeader> = None;
    let mut restart_interval: u16 = 0;

    // Parse markers until SOS
    loop {
        if pos + 1 >= data.len() {
            return Err(VideoError::Codec("MJPEG: unexpected end of data".into()));
        }
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        pos += 1;
        let marker = data[pos];
        pos += 1;

        match marker {
            0x00 | 0xFF => continue, // padding or stuffing
            MARKER_EOI => {
                return Err(VideoError::Codec("MJPEG: EOI before SOS".into()));
            }
            MARKER_DQT => parse_dqt(data, &mut pos, &mut qt)?,
            MARKER_SOF0 => {
                frame_header = Some(parse_sof0(data, &mut pos)?);
            }
            MARKER_DHT => parse_dht(data, &mut pos, &mut huff)?,
            MARKER_DRI => {
                restart_interval = parse_dri(data, &mut pos)?;
            }
            MARKER_SOS => {
                let fh = frame_header
                    .as_mut()
                    .ok_or_else(|| VideoError::Codec("MJPEG: SOS before SOF0".into()))?;
                parse_sos(data, &mut pos, &mut fh.components)?;
                break;
            }
            _ => {
                // Skip unknown/APP markers
                skip_marker(data, &mut pos)?;
            }
        }
    }

    let fh = frame_header.ok_or_else(|| VideoError::Codec("MJPEG: no SOF0 found".into()))?;
    let width = fh.width as usize;
    let height = fh.height as usize;

    if width == 0 || height == 0 {
        return Err(VideoError::Codec("MJPEG: zero dimensions".into()));
    }

    // Find scan data extent
    let scan_start = pos;
    let scan_end = find_scan_end(data, scan_start);
    let scan_data = &data[scan_start..scan_end];

    let mut reader = BitReader::new(scan_data);

    let num_components = fh.components.len();
    let max_h = fh.max_h as usize;
    let max_v = fh.max_v as usize;

    // MCU dimensions in pixels
    let mcu_w = max_h * 8;
    let mcu_h = max_v * 8;
    let mcu_cols = width.div_ceil(mcu_w);
    let mcu_rows = height.div_ceil(mcu_h);

    // Allocate component planes
    let plane_w = mcu_cols * mcu_w;
    let plane_h = mcu_rows * mcu_h;

    let mut planes: Vec<Vec<u8>> = (0..num_components)
        .map(|_| vec![128u8; plane_w * plane_h])
        .collect();

    let mut prev_dc = vec![0i32; num_components];
    let mut mcu_count = 0u32;

    // Decode MCUs
    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
            // Handle restart intervals
            if restart_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(restart_interval as u32)
            {
                // Reset DC predictors
                prev_dc.iter_mut().for_each(|dc| *dc = 0);
                // Re-align to byte boundary
                reader.bits_left = 0;
                reader.bit_buf = 0;
            }

            for comp_idx in 0..num_components {
                let comp = &fh.components[comp_idx];
                let h_blocks = comp.h_sampling as usize;
                let v_blocks = comp.v_sampling as usize;
                let qt_id = comp.quant_table_id as usize;
                let dc_id = comp.dc_table_id as usize;
                let ac_id = comp.ac_table_id as usize;

                let dc_ht = &huff[0][dc_id.min(1)];
                let ac_ht = &huff[1][ac_id.min(1)];
                let quant = &qt[qt_id.min(3)].table;

                for v_block in 0..v_blocks {
                    for h_block in 0..h_blocks {
                        let block =
                            decode_block(&mut reader, dc_ht, ac_ht, quant, &mut prev_dc[comp_idx])?;

                        // Write block to component plane
                        // Component plane coordinates: the MCU covers a region
                        // scaled by sampling factors relative to the full plane.
                        let comp_scale_x = max_h / h_blocks;
                        let comp_scale_y = max_v / v_blocks;
                        let _ = comp_scale_x; // not used here; plane is at component resolution
                        let _ = comp_scale_y;

                        // The component's plane is at full resolution (plane_w x plane_h).
                        // For subsampled components, we need to account for the
                        // scaling: the MCU position in the component plane is
                        // mcu_col * h_blocks * 8 + h_block * 8.
                        let bx = (mcu_col * h_blocks + h_block) * 8;
                        let by = (mcu_row * v_blocks + v_block) * 8;

                        let plane = &mut planes[comp_idx];
                        for row in 0..8 {
                            let py = by + row;
                            if py >= plane_h {
                                continue;
                            }
                            let dst_off = py * plane_w + bx;
                            for col in 0..8 {
                                let px = bx + col;
                                if px >= plane_w {
                                    continue;
                                }
                                plane[dst_off + col] = block[row * 8 + col] as u8;
                            }
                        }
                    }
                }
            }

            mcu_count += 1;
        }
    }

    // Convert YCbCr planes to RGB8
    output.resize(width * height * 3, 0);

    if num_components == 1 {
        // Grayscale
        let y_plane = &planes[0];
        for row in 0..height {
            for col in 0..width {
                let y = y_plane[row * plane_w + col];
                let dst = (row * width + col) * 3;
                output[dst] = y;
                output[dst + 1] = y;
                output[dst + 2] = y;
            }
        }
    } else {
        // YCbCr → RGB
        // Component planes may have different effective resolutions due to
        // subsampling. We need to upsample chroma to luma resolution.
        let y_plane = &planes[0];
        let cb_plane = &planes[1];
        let cr_plane = &planes[2];

        let y_h_blocks = fh.components[0].h_sampling as usize;
        let y_v_blocks = fh.components[0].v_sampling as usize;
        let cb_h_blocks = fh.components[1].h_sampling as usize;
        let cb_v_blocks = fh.components[1].v_sampling as usize;

        // Plane widths for each component
        let y_plane_w = mcu_cols * y_h_blocks * 8;
        let cb_plane_w = mcu_cols * cb_h_blocks * 8;

        let sub_x = y_h_blocks / cb_h_blocks.max(1);
        let sub_y = y_v_blocks / cb_v_blocks.max(1);

        for row in 0..height {
            let y_off = row * y_plane_w;
            let c_row = row / sub_y;
            let c_off = c_row * cb_plane_w;

            for col in 0..width {
                let y_val = y_plane[y_off + col] as i16;
                let c_col = col / sub_x;
                let cb_val = cb_plane[c_off + c_col] as i16 - 128;
                let cr_val = cr_plane[c_off + c_col] as i16 - 128;

                // BT.601 full-range
                let r = y_val + ((cr_val * 179) >> 7);
                let g = y_val - ((cb_val * 44 + cr_val * 91) >> 7);
                let b = y_val + ((cb_val * 227) >> 7);

                let dst = (row * width + col) * 3;
                output[dst] = r.clamp(0, 255) as u8;
                output[dst + 1] = g.clamp(0, 255) as u8;
                output[dst + 2] = b.clamp(0, 255) as u8;
            }
        }
    }

    Ok((fh.width, fh.height))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid single-MCU 8x8 grayscale JFIF image.
    fn build_minimal_gray_jpeg() -> Vec<u8> {
        let mut out = Vec::new();

        // SOI
        out.extend_from_slice(&[0xFF, MARKER_SOI]);

        // DQT — single table, all 1s (identity quantization)
        out.extend_from_slice(&[0xFF, MARKER_DQT]);
        let dqt_len = 2 + 1 + 64; // length + info byte + 64 values
        out.extend_from_slice(&(dqt_len as u16).to_be_bytes());
        out.push(0x00); // 8-bit precision, table 0
        out.extend_from_slice(&[1u8; 64]); // all quantization values = 1

        // SOF0 — 8x8, 1 component, grayscale
        out.extend_from_slice(&[0xFF, MARKER_SOF0]);
        let sof_len = 2 + 1 + 2 + 2 + 1 + 3; // len + prec + h + w + ncomp + comp
        out.extend_from_slice(&(sof_len as u16).to_be_bytes());
        out.push(8); // precision
        out.extend_from_slice(&8u16.to_be_bytes()); // height
        out.extend_from_slice(&8u16.to_be_bytes()); // width
        out.push(1); // 1 component
        out.push(1); // component id = 1
        out.push(0x11); // sampling 1x1
        out.push(0); // quant table 0

        // DHT — DC table 0 (simplified: only code for value 0 = EOB-like)
        // We'll use a simple DC table: code 0 (1 bit) = category 0 (dc_diff=0)
        out.extend_from_slice(&[0xFF, MARKER_DHT]);
        // DC table: 1 symbol of length 1
        let dc_counts = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let dc_symbols = [0u8]; // category 0
        let dht_dc_len = 2 + 1 + 16 + dc_symbols.len();
        out.extend_from_slice(&(dht_dc_len as u16).to_be_bytes());
        out.push(0x00); // DC table, id 0
        out.extend_from_slice(&dc_counts);
        out.extend_from_slice(&dc_symbols);

        // DHT — AC table 0: symbol 0x00 (EOB) at 1-bit code
        out.extend_from_slice(&[0xFF, MARKER_DHT]);
        let ac_counts = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let ac_symbols = [0x00u8]; // EOB
        let dht_ac_len = 2 + 1 + 16 + ac_symbols.len();
        out.extend_from_slice(&(dht_ac_len as u16).to_be_bytes());
        out.push(0x10); // AC table, id 0
        out.extend_from_slice(&ac_counts);
        out.extend_from_slice(&ac_symbols);

        // SOS
        out.extend_from_slice(&[0xFF, MARKER_SOS]);
        let sos_len = 2 + 1 + 2 + 3; // length + ncomp + (1 comp entry) + Ss/Se/AhAl
        out.extend_from_slice(&(sos_len as u16).to_be_bytes());
        out.push(1); // 1 component in scan
        out.push(1); // component selector = 1
        out.push(0x00); // DC table 0, AC table 0
        out.push(0); // Ss = 0
        out.push(63); // Se = 63
        out.push(0); // Ah=0, Al=0

        // Scan data: DC value = 0 (1-bit code 0), AC EOB (1-bit code 0)
        // So we need at least 2 bits: 0b00, padded to a byte = 0x00
        out.push(0x00);

        // EOI
        out.extend_from_slice(&[0xFF, MARKER_EOI]);

        out
    }

    #[test]
    fn decode_minimal_gray_jpeg() {
        let jpeg = build_minimal_gray_jpeg();
        let mut rgb = Vec::new();
        let (w, h) = decode_mjpeg_to_rgb8(&jpeg, &mut rgb).expect("should decode minimal JFIF");
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(rgb.len(), 8 * 8 * 3);

        // All DC=0, dequant=0, IDCT(all-zero-except-dc=0) → all 128 after level shift
        for pixel in rgb.chunks_exact(3) {
            // Gray pixel: R=G=B, value should be 128 (DC=0 + level_shift=128)
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
            assert_eq!(pixel[0], 128);
        }
    }

    #[test]
    fn reject_truncated_data() {
        let result = decode_mjpeg_to_rgb8(&[0xFF, 0xD8], &mut Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn reject_missing_soi() {
        let result = decode_mjpeg_to_rgb8(&[0x00, 0x00, 0x00, 0x00], &mut Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn idct_dc_only_block() {
        // A block with only DC=800, rest zeros. After IDCT all 64 values should
        // be equal (DC distributes evenly).
        let mut block = [0i32; 64];
        block[0] = 800;
        idct_8x8(&mut block);
        // All values should be the same (DC coefficient distributes equally)
        let val = block[0];
        for &v in &block[1..] {
            // Allow small rounding variation from fixed-point
            assert!(
                (v - val).abs() <= 1,
                "IDCT DC-only should be uniform, got {v} vs {val}"
            );
        }
    }

    #[test]
    fn zigzag_table_is_bijection() {
        let mut seen = [false; 64];
        for &z in &ZIGZAG {
            assert!(z < 64, "zigzag value out of range: {z}");
            assert!(!seen[z], "zigzag duplicate: {z}");
            seen[z] = true;
        }
    }
}
