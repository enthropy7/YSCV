//! Bitstream reader for H.264/HEVC Exp-Golomb and bit-level parsing.

use crate::VideoError;

// ---------------------------------------------------------------------------
// Bitstream reader (bit-level access for Exp-Golomb / SPS / PPS parsing)
// ---------------------------------------------------------------------------

/// Reads individual bits and Exp-Golomb coded integers from a byte slice.
pub struct BitstreamReader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) byte_offset: usize,
    pub(crate) bit_offset: u8, // 0..8, bits consumed in current byte
}

impl<'a> BitstreamReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    /// Returns the number of bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.byte_offset >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_offset) * 8 - self.bit_offset as usize
    }

    /// Returns the total number of bits consumed so far.
    pub fn bits_consumed(&self) -> usize {
        self.byte_offset * 8 + self.bit_offset as usize
    }

    /// Reads a single bit (0 or 1).
    pub fn read_bit(&mut self) -> Result<u8, VideoError> {
        if self.byte_offset >= self.data.len() {
            return Err(VideoError::Codec("bitstream exhausted".into()));
        }
        let bit = (self.data[self.byte_offset] >> (7 - self.bit_offset)) & 1;
        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        Ok(bit)
    }

    /// Reads `n` bits as a u32 (MSB first), n <= 32.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, VideoError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(VideoError::Codec(format!(
                "read_bits: requested {n} bits, max is 32"
            )));
        }
        let mut value = 0u32;
        let mut remaining = n;

        // Fast path: consume remaining bits in current byte
        if self.bit_offset > 0 && self.byte_offset < self.data.len() {
            let avail = 8 - self.bit_offset;
            let take = remaining.min(avail);
            let byte = self.data[self.byte_offset];
            let shift = avail - take;
            let mask = (1u8 << take) - 1;
            value = ((byte >> shift) & mask) as u32;
            self.bit_offset += take;
            if self.bit_offset >= 8 {
                self.bit_offset = 0;
                self.byte_offset += 1;
            }
            remaining -= take;
        }

        // Fast path: consume full bytes
        while remaining >= 8 && self.byte_offset < self.data.len() {
            value = (value << 8) | self.data[self.byte_offset] as u32;
            self.byte_offset += 1;
            remaining -= 8;
        }

        // Remainder: bit-by-bit (0-7 bits)
        for _ in 0..remaining {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Ok(value)
    }

    /// Reads an unsigned Exp-Golomb coded integer (ue(v)).
    pub fn read_ue(&mut self) -> Result<u32, VideoError> {
        // Fast path: peek current byte to count leading zeros
        let mut leading_zeros = 0u32;
        loop {
            if self.byte_offset >= self.data.len() {
                return Err(VideoError::Codec("bitstream exhausted".into()));
            }
            let byte = self.data[self.byte_offset];
            let remaining_bits = 8 - self.bit_offset;
            // Mask out already-consumed bits
            let masked = byte << self.bit_offset;
            if masked != 0 {
                // Found a 1 bit — count leading zeros in the masked byte
                // masked is a u8 shifted left by bit_offset, so top bit_offset bits are already consumed
                let lz = (masked as u32).leading_zeros() - 24; // leading_zeros for u8 range
                leading_zeros += lz;
                // Consume lz zeros + 1 (the '1' bit)
                let consume = (lz + 1) as u8;
                self.bit_offset += consume;
                if self.bit_offset >= 8 {
                    self.byte_offset += (self.bit_offset / 8) as usize;
                    self.bit_offset %= 8;
                }
                break;
            }
            // Entire remaining byte is zeros
            leading_zeros += remaining_bits as u32;
            self.bit_offset = 0;
            self.byte_offset += 1;
            if leading_zeros > 31 {
                return Err(VideoError::Codec("exp-golomb overflow".into()));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Ok((1 << leading_zeros) - 1 + suffix)
    }

    /// Reads a signed Exp-Golomb coded integer (se(v)).
    pub fn read_se(&mut self) -> Result<i32, VideoError> {
        let code = self.read_ue()?;
        let value = code.div_ceil(2) as i32;
        if code % 2 == 0 { Ok(-value) } else { Ok(value) }
    }

    /// Skips `n` bits.
    pub fn skip_bits(&mut self, n: usize) -> Result<(), VideoError> {
        let total_bit = self.byte_offset * 8 + self.bit_offset as usize + n;
        let new_byte = total_bit / 8;
        let new_bit = (total_bit % 8) as u8;
        if new_byte > self.data.len() || (new_byte == self.data.len() && new_bit > 0) {
            return Err(VideoError::Codec("bitstream exhausted in skip".into()));
        }
        self.byte_offset = new_byte;
        self.bit_offset = new_bit;
        Ok(())
    }
}
