use yscv_tensor::Tensor;

pub(crate) fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

pub(crate) fn same_shape_data(lhs: &Tensor, rhs: &Tensor) -> Option<usize> {
    if lhs.shape() == rhs.shape() {
        Some(lhs.data().len())
    } else {
        None
    }
}

// ── Backend trait implementation ───────────────────────────────────
pub(crate) fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exponent == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let shift = -1 - unbiased;
        let subnormal = ((mantissa | 0x00800000) >> (shift + 13)) as u16;
        return sign | subnormal;
    }
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    let fp16_exp = ((unbiased + 15) as u16) << 10;
    let fp16_man = (mantissa >> 13) as u16;
    sign | fp16_exp | fp16_man
}

pub(crate) fn f16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = (half & 0x03FF) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 0i32;
        let mut m = mantissa;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_man = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }
    if exponent == 31 {
        let f32_bits = sign | 0x7F800000 | if mantissa != 0 { 0x00400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }
    let f32_exp = ((exponent as u32) + 112) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}
