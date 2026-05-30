// ============================================================================
// MR=6×16 microkernel + pack_a_mr6 unit tests
// ============================================================================

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod mr6_tests {
    use super::super::pack::pack_a_panel_mr6;
    use super::super::*;

    /// Reference matmul: out = A[m, k] × B[k, n], optionally accumulated.
    fn ref_matmul_accumulate(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        accumulate: bool,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = if accumulate { out[i * n + j] } else { 0.0 };
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    fn fill_ramp(buf: &mut [f32], scale: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * scale - 1.5;
        }
    }

    /// Verify `pack_a_panel_mr6` produces the expected (mc/MR6) × kc × MR6
    /// layout — first 6 elements per k are row 0..5 of A at col `pc+p`.
    #[test]
    fn pack_a_panel_mr6_layout_matches_spec() {
        const M: usize = 6;
        const K: usize = 4;
        let a: Vec<f32> = (0..M * K).map(|i| i as f32).collect();
        let mut packed = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed);
        // For each k column p, packed[p*MR6 + i] == a[i*K + p].
        for p in 0..K {
            for i in 0..MR6 {
                let expected = a[i * K + p];
                assert_eq!(packed[p * MR6 + i], expected, "k={p} row={i}");
            }
        }
    }

    /// Direct call into microkernel_6x16_avx_fma with a 6×8 tile from a
    /// 6×K × K×16 multiply. Verify bitwise-close match vs reference scalar.
    #[test]
    fn microkernel_6x16_matches_reference_no_epilogue() {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        const M: usize = 6;
        const K: usize = 16;
        const N: usize = 16;
        let mut a = vec![0.0f32; M * K];
        let mut b = vec![0.0f32; K * N];
        fill_ramp(&mut a, 0.13);
        fill_ramp(&mut b, 0.19);

        // Reference.
        let mut ref_out = vec![0.0f32; M * N];
        ref_matmul_accumulate(&a, &b, &mut ref_out, M, K, N, false);

        // Pack A MR=6 stride.
        let mut packed_a = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed_a);

        // Pack B as two NR=8 panels (panel_0 = cols 0..8, panel_1 = cols 8..16).
        let mut packed_b = vec![0.0f32; 2 * K * NR];
        for p in 0..K {
            for j in 0..NR {
                packed_b[p * NR + j] = b[p * N + j];
                packed_b[K * NR + p * NR + j] = b[p * N + NR + j];
            }
        }

        let mut out = vec![0.0f32; M * N];
        let epilogue = GemmEpilogue::new(None, Activation::None);
        // SAFETY: buffers sized to [M×N], packed_a to MR6 layout, packed_b
        // to two NR=8 panels. `microkernel_6x16_avx_fma` is gated on
        // fma+avx feature detection (checked above).
        #[allow(unsafe_code)]
        unsafe {
            microkernel_6x16_avx_fma(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                packed_b.as_ptr().add(K * NR),
                out.as_mut_ptr(),
                N,
                K,
                false,
                epilogue,
                true,
                0,
                None,
            );
        }

        for i in 0..M * N {
            let d = (out[i] - ref_out[i]).abs();
            assert!(d < 1e-3, "out[{i}]={} ref={} diff={d}", out[i], ref_out[i]);
        }
    }

    /// Same as above but with bias + Relu activation.
    #[test]
    fn microkernel_6x16_matches_reference_bias_relu() {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        const M: usize = 6;
        const K: usize = 8;
        const N: usize = 16;
        let mut a = vec![0.0f32; M * K];
        let mut b = vec![0.0f32; K * N];
        let mut bias = [0.0f32; N];
        fill_ramp(&mut a, 0.11);
        fill_ramp(&mut b, 0.17);
        for (i, v) in bias.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 3.0;
        }

        // Reference with bias + Relu.
        let mut ref_out = vec![0.0f32; M * N];
        ref_matmul_accumulate(&a, &b, &mut ref_out, M, K, N, false);
        for i in 0..M {
            for j in 0..N {
                let v = ref_out[i * N + j] + bias[j];
                ref_out[i * N + j] = if v > 0.0 { v } else { 0.0 };
            }
        }

        let mut packed_a = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed_a);
        let mut packed_b = vec![0.0f32; 2 * K * NR];
        for p in 0..K {
            for j in 0..NR {
                packed_b[p * NR + j] = b[p * N + j];
                packed_b[K * NR + p * NR + j] = b[p * N + NR + j];
            }
        }

        let mut out = vec![0.0f32; M * N];
        let epilogue = GemmEpilogue::new(Some(bias.as_ptr()), Activation::Relu);
        // SAFETY: see preceding test; bias ptr stays valid for this call.
        #[allow(unsafe_code)]
        unsafe {
            microkernel_6x16_avx_fma(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                packed_b.as_ptr().add(K * NR),
                out.as_mut_ptr(),
                N,
                K,
                false,
                epilogue,
                true,
                0,
                None,
            );
        }

        for i in 0..M * N {
            let d = (out[i] - ref_out[i]).abs();
            assert!(
                d < 1e-3,
                "bias+relu out[{i}]={} ref={} diff={d}",
                out[i],
                ref_out[i]
            );
        }
    }
}

// ============================================================================
// low-k tile correctness tests
// ============================================================================

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod low_k_tile_tests {
    use super::super::*;

    fn ref_matmul_fused(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
        act: Activation,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                if let Some(bv) = bias {
                    s += bv[j];
                }
                if let Some(rv) = residual {
                    s += rv[i * n + j];
                }
                s = match act {
                    Activation::Relu => s.max(0.0),
                    Activation::Silu => s / (1.0 + (-s).exp()),
                    Activation::None => s,
                };
                out[i * n + j] = s;
            }
        }
        out
    }

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 131) as f32) * scale + base;
        }
    }

    fn run_case(
        m: usize,
        k: usize,
        n: usize,
        with_bias: bool,
        with_residual: bool,
        act: Activation,
    ) {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        assert_eq!(m % 4, 0);
        assert_eq!(n % 24, 0);
        assert!(k == 16 || k == 24);

        let mut a = vec![0.0f32; m * k];
        fill_ramp(&mut a, 0.013, -0.3);
        let mut b = vec![0.0f32; k * n];
        fill_ramp(&mut b, 0.017, 0.1);
        let bias = if with_bias {
            let mut bv = vec![0.0f32; n];
            fill_ramp(&mut bv, 0.3, -1.0);
            Some(bv)
        } else {
            None
        };
        let residual = if with_residual {
            let mut rv = vec![0.0f32; m * n];
            fill_ramp(&mut rv, 0.011, -0.2);
            Some(rv)
        } else {
            None
        };

        let mut out = vec![0.0f32; m * n];
        let epilogue = GemmEpilogue {
            bias: bias.as_ref().map(|v| v.as_ptr()),
            residual: residual.as_ref().map(|v| v.as_ptr()),
            activation: act,
        };
        // SAFETY: shape gate satisfied (asserted above), AVX+FMA
        // detected, buffers sized appropriately.
        #[allow(unsafe_code)]
        unsafe {
            low_k_tile_4x24_parallel_fused(&a, &b, &mut out, m, k, n, &epilogue);
        }

        let ref_out = ref_matmul_fused(&a, &b, m, k, n, bias.as_deref(), residual.as_deref(), act);
        let mut max_diff = 0.0f32;
        let mut max_i = 0;
        for i in 0..out.len() {
            let d = (out[i] - ref_out[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
            }
        }
        // SiLU uses `silu_avx_fma` with a fast bit-trick exp whose
        // approximation error is ~0.5-1% of the value; loosen tolerance
        // to 3% for SiLU cases, keep 1e-2 for None / Relu.
        let tol = match act {
            Activation::Silu => 3e-2,
            _ => 1e-2,
        };
        assert!(
            max_diff < tol,
            "low_k_tile diverges m={m} k={k} n={n} bias={with_bias} residual={with_residual} \
             act={act:?}: max diff {max_diff} at {max_i}: tile={} ref={}",
            out[max_i],
            ref_out[max_i],
        );
    }

    #[test]
    fn low_k_tile_k16_n96_small() {
        run_case(4, 16, 96, false, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n96_bias_relu() {
        run_case(256, 16, 96, true, false, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k16_n96_residual() {
        run_case(256, 16, 96, true, true, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k16_n96_large() {
        run_case(1024, 16, 96, true, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k24_n144_bias_relu() {
        run_case(256, 24, 144, true, false, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k24_n144_residual() {
        run_case(256, 24, 144, false, true, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n24_minimal() {
        run_case(64, 16, 24, false, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n48_silu() {
        run_case(64, 16, 48, true, false, Activation::Silu);
    }

    #[test]
    fn shape_gate_accepts_hot_tracker_shapes() {
        assert!(use_low_k_tile_avx_fma(16384, 16, 96));
        assert!(use_low_k_tile_avx_fma(4096, 24, 144));
        assert!(use_low_k_tile_avx_fma(4096, 16, 96));
    }

    #[test]
    fn shape_gate_rejects_small_or_misaligned() {
        assert!(!use_low_k_tile_avx_fma(256, 16, 96)); // work < 1M
        assert!(!use_low_k_tile_avx_fma(4097, 16, 96)); // m%4
        assert!(!use_low_k_tile_avx_fma(4096, 16, 25)); // n%24
        assert!(!use_low_k_tile_avx_fma(4096, 32, 96)); // k not 16/24
        assert!(!use_low_k_tile_avx_fma(4096, 8, 96)); // k not 16/24
    }
}

#[cfg(test)]
mod residual_tail_tests {
    use super::super::*;

    fn ref_matmul_fused(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: &[f32],
        residual: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                s += bias[j];
                s += residual[i * n + j];
                out[i * n + j] = s.max(0.0);
            }
        }
        out
    }

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 113) as f32) * scale + base;
        }
    }

    #[test]
    fn blocked_gemm_residual_handles_m_and_n_tails() {
        // Tail shape: m is not divisible by MR=4, n is not divisible by NR=8.
        // Still passes blocked gate (m>=32, k>=32, n>=16).
        let m = 35usize;
        let k = 40usize;
        let n = 18usize;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut bias = vec![0.0f32; n];
        let mut residual = vec![0.0f32; m * n];
        fill_ramp(&mut a, 0.013, -0.4);
        fill_ramp(&mut b, 0.019, 0.2);
        fill_ramp(&mut bias, 0.11, -1.7);
        fill_ramp(&mut residual, 0.007, -0.3);

        let mut out = vec![0.0f32; m * n];
        let epilogue = GemmEpilogue {
            bias: Some(bias.as_ptr()),
            residual: Some(residual.as_ptr()),
            activation: Activation::Relu,
        };
        blocked_gemm_sequential(&a, &b, &mut out, m, k, n, epilogue, None);

        let reference = ref_matmul_fused(&a, &b, m, k, n, &bias, &residual);
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..out.len() {
            let d = (out[i] - reference[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        assert!(
            max_diff < 1e-2,
            "blocked residual tail mismatch: max diff {max_diff} at {max_idx}, got={} ref={}",
            out[max_idx],
            reference[max_idx],
        );
    }
}
