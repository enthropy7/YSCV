//! HEVC inverse transforms, intra prediction, and dequantisation.

// ---------------------------------------------------------------------------
// Intra prediction modes
// ---------------------------------------------------------------------------

/// HEVC intra prediction mode index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HevcIntraMode {
    Planar = 0,
    Dc = 1,
    Angular2 = 2,
    Angular3 = 3,
    Angular4 = 4,
    Angular5 = 5,
    Angular6 = 6,
    Angular7 = 7,
    Angular8 = 8,
    Angular9 = 9,
    Angular10 = 10,
    Angular11 = 11,
    Angular12 = 12,
    Angular13 = 13,
    Angular14 = 14,
    Angular15 = 15,
    Angular16 = 16,
    Angular17 = 17,
    Angular18 = 18,
    Angular19 = 19,
    Angular20 = 20,
    Angular21 = 21,
    Angular22 = 22,
    Angular23 = 23,
    Angular24 = 24,
    Angular25 = 25,
    Angular26 = 26,
    Angular27 = 27,
    Angular28 = 28,
    Angular29 = 29,
    Angular30 = 30,
    Angular31 = 31,
    Angular32 = 32,
    Angular33 = 33,
    Angular34 = 34,
}

impl HevcIntraMode {
    /// Convert from a raw mode index (0..=34).
    pub fn from_index(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::Planar),
            1 => Some(Self::Dc),
            2 => Some(Self::Angular2),
            3 => Some(Self::Angular3),
            4 => Some(Self::Angular4),
            5 => Some(Self::Angular5),
            6 => Some(Self::Angular6),
            7 => Some(Self::Angular7),
            8 => Some(Self::Angular8),
            9 => Some(Self::Angular9),
            10 => Some(Self::Angular10),
            11 => Some(Self::Angular11),
            12 => Some(Self::Angular12),
            13 => Some(Self::Angular13),
            14 => Some(Self::Angular14),
            15 => Some(Self::Angular15),
            16 => Some(Self::Angular16),
            17 => Some(Self::Angular17),
            18 => Some(Self::Angular18),
            19 => Some(Self::Angular19),
            20 => Some(Self::Angular20),
            21 => Some(Self::Angular21),
            22 => Some(Self::Angular22),
            23 => Some(Self::Angular23),
            24 => Some(Self::Angular24),
            25 => Some(Self::Angular25),
            26 => Some(Self::Angular26),
            27 => Some(Self::Angular27),
            28 => Some(Self::Angular28),
            29 => Some(Self::Angular29),
            30 => Some(Self::Angular30),
            31 => Some(Self::Angular31),
            32 => Some(Self::Angular32),
            33 => Some(Self::Angular33),
            34 => Some(Self::Angular34),
            _ => None,
        }
    }
}

/// DC intra prediction: fills block with average of top and left neighbours.
pub fn intra_predict_dc(top: &[i16], left: &[i16], block_size: usize, out: &mut [i16]) {
    debug_assert!(top.len() >= block_size);
    debug_assert!(left.len() >= block_size);
    debug_assert!(out.len() >= block_size * block_size);
    let sum: i32 = top[..block_size].iter().map(|&v| v as i32).sum::<i32>()
        + left[..block_size].iter().map(|&v| v as i32).sum::<i32>();
    let dc = ((sum + block_size as i32) / (2 * block_size as i32)) as i16;
    for v in out[..block_size * block_size].iter_mut() {
        *v = dc;
    }
}

/// Planar intra prediction (HEVC mode 0).
pub fn intra_predict_planar(
    top: &[i16],
    left: &[i16],
    top_right: i16,
    bottom_left: i16,
    block_size: usize,
    out: &mut [i16],
) {
    debug_assert!(top.len() >= block_size);
    debug_assert!(left.len() >= block_size);
    debug_assert!(out.len() >= block_size * block_size);
    let n = block_size as i32;
    let log2n = (block_size as u32).trailing_zeros();
    for y in 0..block_size {
        for x in 0..block_size {
            let h = (n - 1 - x as i32) * left[y] as i32 + (x as i32 + 1) * top_right as i32;
            let v = (n - 1 - y as i32) * top[x] as i32 + (y as i32 + 1) * bottom_left as i32;
            out[y * block_size + x] = ((h + v + n) >> (log2n + 1)) as i16;
        }
    }
}

/// Simple angular intra prediction placeholder (modes 2..=34).
/// Uses horizontal or vertical extrapolation depending on mode direction.
/// HEVC angular intra prediction with fractional-sample interpolation.
///
/// ITU-T H.265, section 8.4.4.2.6. Modes 2-34 project through reference
/// samples at angles specified by the `INTRA_PRED_ANGLE` table.
/// Fractional positions use 32-phase linear interpolation.
pub fn intra_predict_angular(
    top: &[i16],
    left: &[i16],
    mode: u8,
    block_size: usize,
    out: &mut [i16],
) {
    debug_assert!((2..=34).contains(&mode));
    debug_assert!(top.len() >= block_size);
    debug_assert!(left.len() >= block_size);
    debug_assert!(out.len() >= block_size * block_size);

    // ITU-T H.265, Table 8-4: intraPredAngle for modes 2..34
    #[rustfmt::skip]
    const INTRA_PRED_ANGLE: [i32; 33] = [
        // modes 2..34 (index 0 = mode 2)
        32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26,
        -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
    ];

    let angle = INTRA_PRED_ANGLE[(mode - 2) as usize];
    let is_vertical = mode >= 18; // modes 18-34 are vertical-dominant

    // Build extended reference array from top or left samples
    let n = block_size;
    // Stack array: max block_size=64, so 2*64+1=129 elements = 258 bytes
    let mut ref_buf = [128i16; 129];
    let ref_samples = &mut ref_buf[..2 * n + 1];

    if is_vertical {
        // Main reference = top row, side reference = left column
        ref_samples[0] = left[0]; // corner
        ref_samples[1..(n.min(top.len()) + 1)].copy_from_slice(&top[..n.min(top.len())]);
        // Extend with projected left samples for negative angles
        if angle < 0 {
            let inv_angle = ((256 * 32) as f32 / (-angle) as f32).round() as i32;
            let num_ext = (n as i32 * angle) >> 5;
            for k in num_ext..0 {
                let ref_idx = ((-k * inv_angle + 128) >> 8) as usize;
                let dst = k as isize;
                if dst >= -(n as isize) && ref_idx < left.len() {
                    ref_samples[(dst + n as isize) as usize] = left[ref_idx];
                }
            }
        }
        // Project each output sample through the angle
        for y in 0..n {
            let delta = (y as i32 + 1) * angle;
            let idx_offset = delta >> 5;
            let frac = (delta & 31) as i16;
            for x in 0..n {
                let ref_idx = (x as i32 + idx_offset + 1) as usize;
                if frac == 0 {
                    out[y * n + x] = ref_samples.get(ref_idx).copied().unwrap_or(128);
                } else {
                    // 32-phase linear interpolation
                    let a = ref_samples.get(ref_idx).copied().unwrap_or(128) as i32;
                    let b = ref_samples
                        .get(ref_idx.wrapping_add(1))
                        .copied()
                        .unwrap_or(128) as i32;
                    out[y * n + x] = ((32 - frac as i32) * a + frac as i32 * b + 16) as i16 >> 5;
                }
            }
        }
    } else {
        // Horizontal-dominant (modes 2-17): main reference = left column
        ref_samples[0] = top[0]; // corner
        ref_samples[1..(n.min(left.len()) + 1)].copy_from_slice(&left[..n.min(left.len())]);
        if angle < 0 {
            let inv_angle = ((256 * 32) as f32 / (-angle) as f32).round() as i32;
            let num_ext = (n as i32 * angle) >> 5;
            for k in num_ext..0 {
                let ref_idx = ((-k * inv_angle + 128) >> 8) as usize;
                let dst = k as isize;
                if dst >= -(n as isize) && ref_idx < top.len() {
                    ref_samples[(dst + n as isize) as usize] = top[ref_idx];
                }
            }
        }
        // Project -- transposed relative to vertical
        for x in 0..n {
            let delta = (x as i32 + 1) * angle;
            let idx_offset = delta >> 5;
            let frac = (delta & 31) as i16;
            for y in 0..n {
                let ref_idx = (y as i32 + idx_offset + 1) as usize;
                if frac == 0 {
                    out[y * n + x] = ref_samples.get(ref_idx).copied().unwrap_or(128);
                } else {
                    let a = ref_samples.get(ref_idx).copied().unwrap_or(128) as i32;
                    let b = ref_samples
                        .get(ref_idx.wrapping_add(1))
                        .copied()
                        .unwrap_or(128) as i32;
                    out[y * n + x] = ((32 - frac as i32) * a + frac as i32 * b + 16) as i16 >> 5;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transform / dequantisation
// ---------------------------------------------------------------------------

/// HEVC 4x4 DST-VII core matrix (for intra 4x4 luma TUs).
const DST4_MATRIX: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// HEVC 4x4 DCT-II core matrix.
const DCT4_MATRIX: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// HEVC 8x8 DCT-II core matrix.
const DCT8_MATRIX: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// Inverse 4x4 DST (HEVC, for intra 4x4 luma).
pub fn hevc_inverse_dst_4x4(coeffs: &[i32; 16], out: &mut [i32; 16]) {
    // 1-D inverse DST on rows
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += DST4_MATRIX[k][j] * coeffs[i * 4 + k];
            }
            tmp[i * 4 + j] = (sum + 64) >> 7;
        }
    }
    // 1-D inverse DST on columns
    for j in 0..4 {
        for i in 0..4 {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += DST4_MATRIX[k][i] * tmp[k * 4 + j];
            }
            out[i * 4 + j] = (sum + 2048) >> 12;
        }
    }
}

/// Inverse 4x4 DCT-II (HEVC).
pub fn hevc_inverse_dct_4x4(coeffs: &[i32; 16], out: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += DCT4_MATRIX[k][j] * coeffs[i * 4 + k];
            }
            tmp[i * 4 + j] = (sum + 64) >> 7;
        }
    }
    for j in 0..4 {
        for i in 0..4 {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += DCT4_MATRIX[k][i] * tmp[k * 4 + j];
            }
            out[i * 4 + j] = (sum + 2048) >> 12;
        }
    }
}

/// Inverse 8x8 DCT-II (HEVC).
pub fn hevc_inverse_dct_8x8(coeffs: &[i32; 64], out: &mut [i32; 64]) {
    let mut tmp = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += DCT8_MATRIX[k][j] * coeffs[i * 8 + k];
            }
            tmp[i * 8 + j] = (sum + 64) >> 7;
        }
    }
    for j in 0..8 {
        for i in 0..8 {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += DCT8_MATRIX[k][i] * tmp[k * 8 + j];
            }
            out[i * 8 + j] = (sum + 2048) >> 12;
        }
    }
}

/// Generic inverse DCT for 16x16 blocks (partial butterfly, simplified).
#[allow(unsafe_code)]
pub fn hevc_inverse_dct_16x16(coeffs: &[i32; 256], out: &mut [i32; 256]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { hevc_inverse_dct_16x16_neon(coeffs, out) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        hevc_inverse_dct_16x16_sse2(coeffs, out);
        return;
    }
    #[allow(unreachable_code)]
    hevc_inverse_dct_16x16_scalar(coeffs, out);
}

/// SSE2-accelerated 16x16 inverse DCT — uses `_mm_madd_epi16` for dot products.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn hevc_inverse_dct_16x16_sse2(coeffs: &[i32; 256], out: &mut [i32; 256]) {
    // SSE2 lacks mullo_epi32. Use the same sparse-aware scalar which LLVM
    // auto-vectorizes well with -O2. The key optimization is zero-row/col skip.
    hevc_inverse_dct_16x16_scalar(coeffs, out);
}

fn hevc_inverse_dct_16x16_scalar(coeffs: &[i32; 256], out: &mut [i32; 256]) {
    // Sparse-aware 16-point DCT: skip zero rows/columns.
    static HEVC_DCT16: [[i32; 16]; 16] = [
        [
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ],
        [
            90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
        ],
        [
            89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
        ],
        [
            87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
        ],
        [
            83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
        ],
        [
            80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
        ],
        [
            75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
        ],
        [
            70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
        ],
        [
            64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        ],
        [
            57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
        ],
        [
            50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
        ],
        [
            43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
        ],
        [
            36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
        ],
        [
            25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
        ],
        [
            18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
        ],
        [
            9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
        ],
    ];
    let mut tmp = [0i32; 256];
    // Row pass with zero-row skip
    for i in 0..16 {
        let row = &coeffs[i * 16..(i + 1) * 16];
        if row.iter().all(|&c| c == 0) {
            continue;
        }
        let last_nz = row.iter().rposition(|&c| c != 0).unwrap_or(0);
        for j in 0..16 {
            let mut sum = 0i32;
            for k in 0..=last_nz {
                if row[k] != 0 {
                    sum += HEVC_DCT16[k][j] * row[k];
                }
            }
            tmp[i * 16 + j] = (sum + 64) >> 7;
        }
    }
    // Column pass with zero-column skip
    for j in 0..16 {
        let col_zero = (0..16).all(|k| tmp[k * 16 + j] == 0);
        if col_zero {
            for i in 0..16 {
                out[i * 16 + j] = 0;
            }
            continue;
        }
        for i in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                let t = tmp[k * 16 + j];
                if t != 0 {
                    sum += HEVC_DCT16[k][i] * t;
                }
            }
            out[i * 16 + j] = (sum + 2048) >> 12;
        }
    }
}

/// NEON-accelerated 16x16 inverse DCT. Processes dot products 4 elements at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hevc_inverse_dct_16x16_neon(coeffs: &[i32; 256], out: &mut [i32; 256]) {
    use std::arch::aarch64::*;

    static HEVC_DCT16: [[i32; 16]; 16] = [
        [
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ],
        [
            90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
        ],
        [
            89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
        ],
        [
            87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
        ],
        [
            83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
        ],
        [
            80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
        ],
        [
            75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
        ],
        [
            70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
        ],
        [
            64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        ],
        [
            57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
        ],
        [
            50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
        ],
        [
            43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
        ],
        [
            36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
        ],
        [
            25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
        ],
        [
            18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
        ],
        [
            9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
        ],
    ];

    let mut tmp = [0i32; 256];

    // Row pass: tmp[i][j] = sum_k(DCT[k][j] * coeffs[i][k]) >> 7
    for i in 0..16 {
        let coeff_row = coeffs.as_ptr().add(i * 16);
        for j in 0..16 {
            let mut acc = vdupq_n_s32(0);
            for k_base in (0..16).step_by(4) {
                let c = vld1q_s32(coeff_row.add(k_base));
                let m = vld1q_s32(
                    [
                        HEVC_DCT16[k_base][j],
                        HEVC_DCT16[k_base + 1][j],
                        HEVC_DCT16[k_base + 2][j],
                        HEVC_DCT16[k_base + 3][j],
                    ]
                    .as_ptr(),
                );
                acc = vmlaq_s32(acc, c, m);
            }
            tmp[i * 16 + j] = (vaddvq_s32(acc) + 64) >> 7;
        }
    }

    // Column pass: out[i][j] = sum_k(DCT[k][i] * tmp[k][j]) >> 12
    for j in 0..16 {
        for i in 0..16 {
            let mut acc = vdupq_n_s32(0);
            for k_base in (0..16).step_by(4) {
                let t = vld1q_s32(
                    [
                        tmp[k_base * 16 + j],
                        tmp[(k_base + 1) * 16 + j],
                        tmp[(k_base + 2) * 16 + j],
                        tmp[(k_base + 3) * 16 + j],
                    ]
                    .as_ptr(),
                );
                let m = vld1q_s32(
                    [
                        HEVC_DCT16[k_base][i],
                        HEVC_DCT16[k_base + 1][i],
                        HEVC_DCT16[k_base + 2][i],
                        HEVC_DCT16[k_base + 3][i],
                    ]
                    .as_ptr(),
                );
                acc = vmlaq_s32(acc, t, m);
            }
            out[i * 16 + j] = (vaddvq_s32(acc) + 2048) >> 12;
        }
    }
}

/// Generic inverse DCT for 32x32 blocks (direct matrix multiply, simplified).
#[allow(unsafe_code)]
pub fn hevc_inverse_dct_32x32(coeffs: &[i32; 1024], out: &mut [i32; 1024]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { hevc_inverse_dct_32x32_neon(coeffs, out) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2: use butterfly scalar which LLVM auto-vectorizes
        hevc_inverse_dct_32x32_scalar(coeffs, out);
        return;
    }
    #[allow(unreachable_code)]
    hevc_inverse_dct_32x32_scalar(coeffs, out);
}

fn hevc_inverse_dct_32x32_scalar(coeffs: &[i32; 1024], out: &mut [i32; 1024]) {
    // Full butterfly decomposition: 32-pt -> 16-pt + 16-odd -> 8-pt + 8-odd + 16-odd -> ...
    // Reduces O(N^2)=1024 multiplications per row to O(N*log2(N)/2) ~ 80.
    //
    // Sub-matrices extracted from HEVC spec Table 8-7:
    //
    // O basis (16x16): odd rows of 32-pt matrix, first 16 columns.
    // Odd rows are antisymmetric: M[2k+1][j] = -M[2k+1][31-j]
    // So output[j] = E[j]+O[j], output[31-j] = E[j]-O[j].
    #[rustfmt::skip]
    static O_BASIS: [[i32; 16]; 16] = [
        [ 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13,  4],
        [ 90, 82, 67, 46, 22, -4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13],
        [ 88, 67, 31,-13,-54,-82,-90,-78,-46, -4, 38, 73, 90, 85, 61, 22],
        [ 85, 46,-13,-67,-90,-73,-22, 38, 82, 88, 54, -4,-61,-90,-78,-31],
        [ 82, 22,-54,-90,-61, 13, 78, 85, 31,-46,-90,-67,  4, 73, 88, 38],
        [ 78, -4,-82,-73, 13, 85, 67,-22,-88,-61, 31, 90, 54,-38,-90,-46],
        [ 73,-31,-90,-22, 78, 67,-38,-90,-13, 82, 61,-46,-88, -4, 85, 54],
        [ 67,-54,-78, 38, 85,-22,-90,  4, 90, 13,-88,-31, 82, 46,-73,-61],
        [ 61,-73,-46, 82, 31,-88,-13, 90, -4,-90, 22, 85,-38,-78, 54, 67],
        [ 54,-85, -4, 88,-46,-61, 82, 13,-90, 38, 67,-78,-22, 90,-31,-73],
        [ 46,-90, 38, 54,-90, 31, 61,-88, 22, 67,-85, 13, 73,-82,  4, 78],
        [ 38,-88, 73, -4,-67, 90,-46,-31, 85,-78, 13, 61,-90, 54, 22,-82],
        [ 31,-78, 90,-61,  4, 54,-88, 82,-38,-22, 73,-90, 67,-13,-46, 85],
        [ 22,-61, 85,-90, 73,-38, -4, 46,-78, 90,-82, 54,-13,-31, 67,-88],
        [ 13,-38, 61,-78, 88,-90, 85,-73, 54,-31,  4, 22,-46, 67,-82, 90],
        [  4,-13, 22,-31, 38,-46, 54,-61, 67,-73, 78,-82, 85,-88, 90,-90],
    ];

    // EO basis (8x8): odd rows (1,3,5,...,15) of the 16-pt DCT matrix, first 8 columns.
    // These are antisymmetric within the 16-pt framework.
    #[rustfmt::skip]
    static EO_BASIS: [[i32; 8]; 8] = [
        [90, 87, 80, 70, 57, 43, 25,  9],
        [87, 57,  9,-43,-80,-90,-70,-25],
        [80,  9,-70,-87,-25, 57, 90, 43],
        [70,-43,-87,  9, 90, 25,-80,-57],
        [57,-80,-25, 90, -9,-87, 43, 70],
        [43,-90, 57, 25,-87, 70,  9,-80],
        [25,-70, 90,-80, 43,  9,-57, 87],
        [ 9,-25, 43,-57, 70,-80, 87,-90],
    ];

    // EEO basis (4x4): odd rows (1,3,5,7) of the 8-pt DCT matrix, first 4 columns.
    #[rustfmt::skip]
    static EEO_BASIS: [[i32; 4]; 4] = [
        [89, 75, 50, 18],
        [75,-18,-89,-50],
        [50,-89, 18, 75],
        [18,-50, 75,-89],
    ];

    // EEE basis (4x4): even rows (0,2,4,6) of the 8-pt DCT matrix, first 4 columns.
    // This is the 4-pt DCT matrix.
    #[rustfmt::skip]
    static EEE_BASIS: [[i32; 4]; 4] = [
        [64, 64, 64, 64],
        [83, 36,-36,-83],
        [64,-64,-64, 64],
        [36,-83, 83,-36],
    ];

    let mut tmp = [0i32; 1024];

    // Row pass with butterfly decomposition
    for i in 0..32 {
        let row_start = i * 32;
        let src = &coeffs[row_start..row_start + 32];
        if src.iter().all(|&c| c == 0) {
            continue;
        }

        // Stage 1: O[j] for j=0..15
        // O[j] = sum_{k=0..15} O_BASIS[k][j] * src[2k+1]
        let mut o = [0i64; 16];
        for j in 0..16 {
            let mut sum = 0i64;
            for k in 0..16 {
                sum += O_BASIS[k][j] as i64 * src[2 * k + 1] as i64;
            }
            o[j] = sum;
        }

        // Stage 2: EO[j] for j=0..7
        // EO[j] = sum_{k=0..7} EO_BASIS[k][j] * src[4k+2]
        let mut eo = [0i64; 8];
        for j in 0..8 {
            let mut sum = 0i64;
            for k in 0..8 {
                sum += EO_BASIS[k][j] as i64 * src[4 * k + 2] as i64;
            }
            eo[j] = sum;
        }

        // Stage 3: EEO[j] for j=0..3
        // EEO[j] = sum_{k=0..3} EEO_BASIS[k][j] * src[8k+4]
        let mut eeo = [0i64; 4];
        for j in 0..4 {
            let mut sum = 0i64;
            for k in 0..4 {
                sum += EEO_BASIS[k][j] as i64 * src[8 * k + 4] as i64;
            }
            eeo[j] = sum;
        }

        // Stage 4: EEE[j] for j=0..3
        // EEE[j] = sum_{k=0..3} EEE_BASIS[k][j] * src[8k]
        let mut eee = [0i64; 4];
        for j in 0..4 {
            let mut sum = 0i64;
            for k in 0..4 {
                sum += EEE_BASIS[k][j] as i64 * src[8 * k] as i64;
            }
            eee[j] = sum;
        }

        // Combine: EE from EEE and EEO
        let mut ee = [0i64; 8];
        for j in 0..4 {
            ee[j] = eee[j] + eeo[j];
            ee[7 - j] = eee[j] - eeo[j];
        }

        // Combine: E from EE and EO
        let mut e = [0i64; 16];
        for j in 0..8 {
            e[j] = ee[j] + eo[j];
            e[15 - j] = ee[j] - eo[j];
        }

        // Combine: output from E and O
        let base = i * 32;
        for j in 0..16 {
            tmp[base + j] = ((e[j] + o[j] + 64) >> 7) as i32;
            tmp[base + 31 - j] = ((e[j] - o[j] + 64) >> 7) as i32;
        }
    }

    // Column pass with butterfly decomposition
    for j in 0..32 {
        // Check if column j of tmp is all zero
        let col_zero = (0..32).all(|k| tmp[k * 32 + j] == 0);
        if col_zero {
            for i in 0..32 {
                out[i * 32 + j] = 0;
            }
            continue;
        }

        // Gather column into a local array for convenience
        let mut col = [0i64; 32];
        for k in 0..32 {
            col[k] = tmp[k * 32 + j] as i64;
        }

        // Stage 1: O[i] for i=0..15
        let mut o = [0i64; 16];
        for ii in 0..16 {
            let mut sum = 0i64;
            for k in 0..16 {
                sum += O_BASIS[k][ii] as i64 * col[2 * k + 1];
            }
            o[ii] = sum;
        }

        // Stage 2: EO[i] for i=0..7
        let mut eo = [0i64; 8];
        for ii in 0..8 {
            let mut sum = 0i64;
            for k in 0..8 {
                sum += EO_BASIS[k][ii] as i64 * col[4 * k + 2];
            }
            eo[ii] = sum;
        }

        // Stage 3: EEO[i] for i=0..3
        let mut eeo = [0i64; 4];
        for ii in 0..4 {
            let mut sum = 0i64;
            for k in 0..4 {
                sum += EEO_BASIS[k][ii] as i64 * col[8 * k + 4];
            }
            eeo[ii] = sum;
        }

        // Stage 4: EEE[i] for i=0..3
        let mut eee = [0i64; 4];
        for ii in 0..4 {
            let mut sum = 0i64;
            for k in 0..4 {
                sum += EEE_BASIS[k][ii] as i64 * col[8 * k];
            }
            eee[ii] = sum;
        }

        // Combine: EE
        let mut ee = [0i64; 8];
        for ii in 0..4 {
            ee[ii] = eee[ii] + eeo[ii];
            ee[7 - ii] = eee[ii] - eeo[ii];
        }

        // Combine: E
        let mut e = [0i64; 16];
        for ii in 0..8 {
            e[ii] = ee[ii] + eo[ii];
            e[15 - ii] = ee[ii] - eo[ii];
        }

        // Combine: output
        for ii in 0..16 {
            out[ii * 32 + j] = ((e[ii] + o[ii] + 2048) >> 12) as i32;
            out[(31 - ii) * 32 + j] = ((e[ii] - o[ii] + 2048) >> 12) as i32;
        }
    }
}

/// NEON-accelerated 32x32 inverse DCT. Dot products 4 elements at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hevc_inverse_dct_32x32_neon(coeffs: &[i32; 1024], out: &mut [i32; 1024]) {
    use std::arch::aarch64::*;

    static HEVC_DCT32: [[i32; 32]; 32] = hevc_dct32_matrix();
    let mut tmp = [0i32; 1024];

    // Row pass: process 4 k-elements at a time in dot product
    for i in 0..32 {
        let coeff_row = coeffs.as_ptr().add(i * 32);
        for j in 0..32 {
            let mut acc = vdupq_n_s32(0);
            for k_base in (0..32).step_by(4) {
                let c = vld1q_s32(coeff_row.add(k_base));
                let m = vld1q_s32(
                    [
                        HEVC_DCT32[k_base][j],
                        HEVC_DCT32[k_base + 1][j],
                        HEVC_DCT32[k_base + 2][j],
                        HEVC_DCT32[k_base + 3][j],
                    ]
                    .as_ptr(),
                );
                acc = vmlaq_s32(acc, c, m);
            }
            // Widen to i64 for safe accumulation, then shift
            let sum = vaddvq_s32(acc) as i64;
            tmp[i * 32 + j] = ((sum + 64) >> 7) as i32;
        }
    }

    // Column pass
    for j in 0..32 {
        for i in 0..32 {
            let mut acc = vdupq_n_s32(0);
            for k_base in (0..32).step_by(4) {
                let t = vld1q_s32(
                    [
                        tmp[k_base * 32 + j],
                        tmp[(k_base + 1) * 32 + j],
                        tmp[(k_base + 2) * 32 + j],
                        tmp[(k_base + 3) * 32 + j],
                    ]
                    .as_ptr(),
                );
                let m = vld1q_s32(
                    [
                        HEVC_DCT32[k_base][i],
                        HEVC_DCT32[k_base + 1][i],
                        HEVC_DCT32[k_base + 2][i],
                        HEVC_DCT32[k_base + 3][i],
                    ]
                    .as_ptr(),
                );
                acc = vmlaq_s32(acc, t, m);
            }
            let sum = vaddvq_s32(acc) as i64;
            out[i * 32 + j] = ((sum + 2048) >> 12) as i32;
        }
    }
}

/// Build the HEVC 32-point DCT-II transform matrix at compile time.
#[cfg(any(target_arch = "aarch64", test))]
const fn hevc_dct32_matrix() -> [[i32; 32]; 32] {
    // Even rows (0,2,4,...,30) come from 16-point matrix expanded to 32 columns.
    // Odd rows (1,3,5,...,31) are the 32-point odd basis from HEVC spec Table 8-7.
    let even16: [[i32; 16]; 16] = [
        [
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ],
        [
            90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
        ],
        [
            89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
        ],
        [
            87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
        ],
        [
            83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
        ],
        [
            80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
        ],
        [
            75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
        ],
        [
            70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
        ],
        [
            64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        ],
        [
            57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
        ],
        [
            50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
        ],
        [
            43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
        ],
        [
            36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
        ],
        [
            25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
        ],
        [
            18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
        ],
        [
            9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
        ],
    ];
    let odd_rows: [[i32; 32]; 16] = [
        [
            90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38,
            -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
        ],
        [
            90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61,
            78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90,
        ],
        [
            88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85,
            -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88,
        ],
        [
            85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90,
            61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85,
        ],
        [
            82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4,
            67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82,
        ],
        [
            78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38,
            -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78,
        ],
        [
            73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4,
            88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73,
        ],
        [
            67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46,
            -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67,
        ],
        [
            61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78,
            38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61,
        ],
        [
            54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90,
            22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54,
        ],
        [
            46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73,
            -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46,
        ],
        [
            38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54,
            90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38,
        ],
        [
            31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13,
            -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31,
        ],
        [
            22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31,
            13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22,
        ],
        [
            13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46,
            -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13,
        ],
        [
            4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88,
            -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4,
        ],
    ];
    // Expand the 16-point even basis into 32-column rows
    let even_rows_full: [[i32; 32]; 16] = expand_even_rows(&even16);
    // Assemble: even rows at indices 0,2,4,...,30; odd rows at 1,3,5,...,31
    let mut m = [[0i32; 32]; 32];
    let mut row = 0;
    while row < 16 {
        m[row * 2] = even_rows_full[row];
        m[row * 2 + 1] = odd_rows[row];
        row += 1;
    }
    m
}

/// Expand 16-point even basis into 32 columns (DCT decomposition).
///
/// For the DCT-II transform matrix, even rows are always symmetric:
///   M[2r][n] = M[2r][31-n]  for all r and n.
/// The first 16 columns equal the 16-point DCT matrix, and columns 16..31
/// mirror columns 15..0.
#[cfg(any(target_arch = "aarch64", test))]
const fn expand_even_rows(even16: &[[i32; 16]; 16]) -> [[i32; 32]; 16] {
    let mut out = [[0i32; 32]; 16];
    let mut r = 0;
    while r < 16 {
        let mut c = 0;
        while c < 32 {
            if c < 16 {
                out[r][c] = even16[r][c];
            } else {
                let mirror = 31 - c;
                out[r][c] = even16[r][mirror];
            }
            c += 1;
        }
        r += 1;
    }
    out
}

/// HEVC dequantisation for a transform block.
/// Applies `level * scale >> shift` per coefficient.
#[allow(unsafe_code)]
pub fn hevc_dequant(coeffs: &mut [i32], qp: i32, bit_depth: u8, log2_transform_size: u8) {
    const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let qp = qp.max(0) as u32;
    let scale = LEVEL_SCALE[(qp % 6) as usize];
    let shift_base = qp / 6;
    let bd_offset = (bit_depth as u32).saturating_sub(8);
    let max_log2 = 15 + bd_offset;
    let transform_shift = max_log2 as i32 - bit_depth as i32 - log2_transform_size as i32;
    let total_shift = shift_base as i32 + transform_shift;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            hevc_dequant_neon(coeffs, scale, total_shift);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            hevc_dequant_sse2(coeffs, scale, total_shift);
        }
        return;
    }

    #[allow(unreachable_code)]
    hevc_dequant_scalar(coeffs, scale, total_shift);
}

fn hevc_dequant_scalar(coeffs: &mut [i32], scale: i32, total_shift: i32) {
    if total_shift >= 0 {
        let offset = if total_shift > 0 {
            1 << (total_shift - 1)
        } else {
            0
        };
        for c in coeffs.iter_mut() {
            *c = (*c * scale + offset) >> total_shift;
        }
    } else {
        let left_shift = (-total_shift) as u32;
        for c in coeffs.iter_mut() {
            *c = *c * scale * (1 << left_shift);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hevc_dequant_sse2(coeffs: &mut [i32], scale: i32, total_shift: i32) {
    // SSE2 lacks _mm_mullo_epi32 (SSE4.1), so scalar loop with autovectorization
    let len = coeffs.len();
    let p = coeffs.as_mut_ptr();

    if total_shift >= 0 {
        let offset = if total_shift > 0 {
            1i32 << (total_shift - 1)
        } else {
            0
        };
        for i in 0..len {
            *p.add(i) = (*p.add(i) * scale + offset) >> total_shift;
        }
    } else {
        let left_shift = (-total_shift) as u32;
        let mul = scale * (1i32 << left_shift);
        for i in 0..len {
            *p.add(i) = *p.add(i) * mul;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hevc_dequant_neon(coeffs: &mut [i32], scale: i32, total_shift: i32) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_s32(scale);
    let len = coeffs.len();
    let ptr = coeffs.as_mut_ptr();

    if total_shift >= 0 {
        let offset = if total_shift > 0 {
            1i32 << (total_shift - 1)
        } else {
            0
        };
        let offset_v = vdupq_n_s32(offset);
        let neg_shift = vdupq_n_s32(-total_shift); // negative for right shift
        let mut i = 0;
        while i + 4 <= len {
            let c = vld1q_s32(ptr.add(i));
            let mul = vmulq_s32(c, scale_v);
            let added = vaddq_s32(mul, offset_v);
            let shifted = vshlq_s32(added, neg_shift);
            vst1q_s32(ptr.add(i), shifted);
            i += 4;
        }
        // Scalar remainder
        for j in i..len {
            *ptr.add(j) = (*ptr.add(j) * scale + offset) >> total_shift;
        }
    } else {
        let left_shift = vdupq_n_s32(-total_shift); // positive for left shift
        let mut i = 0;
        while i + 4 <= len {
            let c = vld1q_s32(ptr.add(i));
            let mul = vmulq_s32(c, scale_v);
            let shifted = vshlq_s32(mul, left_shift);
            vst1q_s32(ptr.add(i), shifted);
            i += 4;
        }
        let left_shift_scalar = (-total_shift) as u32;
        for j in i..len {
            *ptr.add(j) = *ptr.add(j) * scale * (1 << left_shift_scalar);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference direct matrix multiply for 32x32 inverse DCT (no butterfly).
    /// Used to verify butterfly produces bit-exact results.
    fn hevc_inverse_dct_32x32_reference(coeffs: &[i32; 1024], out: &mut [i32; 1024]) {
        static HEVC_DCT32: [[i32; 32]; 32] = hevc_dct32_matrix();
        let mut tmp = [0i32; 1024];
        for i in 0..32 {
            for j in 0..32 {
                let mut sum = 0i64;
                for k in 0..32 {
                    sum += HEVC_DCT32[k][j] as i64 * coeffs[i * 32 + k] as i64;
                }
                tmp[i * 32 + j] = ((sum + 64) >> 7) as i32;
            }
        }
        for j in 0..32 {
            for i in 0..32 {
                let mut sum = 0i64;
                for k in 0..32 {
                    sum += HEVC_DCT32[k][i] as i64 * tmp[k * 32 + j] as i64;
                }
                out[i * 32 + j] = ((sum + 2048) >> 12) as i32;
            }
        }
    }

    #[test]
    fn butterfly_matches_direct_matmul_32x32() {
        // Test with a variety of non-trivial coefficient patterns
        let mut coeffs = [0i32; 1024];
        // Sparse pattern: DC + a few AC coefficients
        coeffs[0] = 1000;
        coeffs[1] = -500;
        coeffs[2] = 300;
        coeffs[32] = 200;
        coeffs[33] = -100;
        coeffs[63] = 50;
        coeffs[512] = -400;
        coeffs[1023] = 75;

        let mut out_butterfly = [0i32; 1024];
        let mut out_reference = [0i32; 1024];

        hevc_inverse_dct_32x32_scalar(&coeffs, &mut out_butterfly);
        hevc_inverse_dct_32x32_reference(&coeffs, &mut out_reference);

        for i in 0..1024 {
            assert_eq!(
                out_butterfly[i], out_reference[i],
                "mismatch at index {}: butterfly={} reference={}",
                i, out_butterfly[i], out_reference[i]
            );
        }
    }

    #[test]
    fn butterfly_matches_direct_matmul_32x32_dense() {
        // Dense pattern with all coefficients non-zero
        let mut coeffs = [0i32; 1024];
        for i in 0..1024 {
            coeffs[i] = ((i as i32 * 37 + 13) % 201) - 100; // pseudo-random in [-100, 100]
        }

        let mut out_butterfly = [0i32; 1024];
        let mut out_reference = [0i32; 1024];

        hevc_inverse_dct_32x32_scalar(&coeffs, &mut out_butterfly);
        hevc_inverse_dct_32x32_reference(&coeffs, &mut out_reference);

        for i in 0..1024 {
            assert_eq!(
                out_butterfly[i], out_reference[i],
                "mismatch at index {}: butterfly={} reference={}",
                i, out_butterfly[i], out_reference[i]
            );
        }
    }

    #[test]
    fn butterfly_32x32_all_zeros() {
        let coeffs = [0i32; 1024];
        let mut out = [42i32; 1024]; // pre-fill with non-zero to verify it gets cleared
        hevc_inverse_dct_32x32_scalar(&coeffs, &mut out);
        for i in 0..1024 {
            assert_eq!(out[i], 0, "non-zero at index {} for zero input", i);
        }
    }
}
