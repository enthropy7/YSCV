use super::*;

pub(super) fn exec_gemm(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let c = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let alpha = get_attr_float(node, "alpha").unwrap_or(1.0);
    let beta_val = get_attr_float(node, "beta").unwrap_or(1.0);
    let trans_a = get_attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;

    // Borrow-or-own pattern: avoid cloning when no transpose needed
    let a_owned;
    let a_ref: &Tensor = if trans_a {
        a_owned = a.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        &a_owned
    } else {
        a
    };
    let b_owned;
    let b_ref: &Tensor = if trans_b {
        b_owned = b.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        &b_owned
    } else {
        b
    };

    let mut out = matmul_2d(a_ref, b_ref).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    if (alpha - 1.0).abs() > f32::EPSILON {
        out = out.scale(alpha);
    }
    if let Some(c_tensor) = c {
        if (beta_val - 1.0).abs() > f32::EPSILON {
            let scaled_c = c_tensor.scale(beta_val);
            out = kernel_add(&out, &scaled_c).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        } else {
            out = kernel_add(&out, c_tensor).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        }
    }
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// Fused `Transpose(perm=[0,2,1]) → MatMul` dispatch. Reads the
/// pre-transpose source directly via a `transA=1` MatMul kernel —
/// avoids materialising the transposed intermediate tensor (which in
/// tracker is 64 KB per branch, shared across cls/reg branches). The
/// Transpose node itself was elided from the execution plan by the
/// loader when it detected this pattern.
///
/// Layout: `Transpose` input is `[batch, K, M]` (rank-3); after perm
/// `[0, 2, 1]` it would become `[batch, M, K]`, which MatMul would
/// treat as the left operand — so the actual matmul is
/// `[batch, M, K] @ [batch, K, N] → [batch, M, N]`. We skip the
/// physical transpose and compute directly with `transA=1`.
pub(super) fn exec_fused_transpose_matmul(
    transpose_node: &OnnxNode,
    matmul_node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let a_pre = get_tensor(env, &transpose_node.name, &transpose_node.inputs[0])?;
    let b = get_tensor(env, &matmul_node.name, &matmul_node.inputs[1])?;

    let a_shape = a_pre.shape();
    let b_shape = b.shape();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    // Pre-transpose layout is `[..., K, M]`. Post-transpose (what the
    // MatMul sees as A) would be `[..., M, K]` with `M = a[-1]` and
    // `K = a[-2]`. `B` is `[..., K, N]` with `N = b[-1]`.
    let m = a_shape[a_rank - 1];
    let k = a_shape[a_rank - 2];
    let n = b_shape[b_rank - 1];

    // Broadcast batch dimensions (same logic as `exec_matmul`). For
    // tracker both branches land at batch_size=1; the general code
    // handles ≥ 1 correctly.
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];
    let max_batch_rank = a_batch.len().max(b_batch.len());
    let mut out_batch = Vec::with_capacity(max_batch_rank);
    for i in 0..max_batch_rank {
        let ad = if i < max_batch_rank - a_batch.len() {
            1
        } else {
            a_batch[i - (max_batch_rank - a_batch.len())]
        };
        let bd = if i < max_batch_rank - b_batch.len() {
            1
        } else {
            b_batch[i - (max_batch_rank - b_batch.len())]
        };
        out_batch.push(ad.max(bd));
    }
    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let a_mat_stride = k * m;
    let b_mat_stride = k * n;
    let out_mat_stride = m * n;
    let a_data = a_pre.data();
    let b_data = b.data();
    let a_batch_total: usize = a_batch.iter().product::<usize>().max(1);
    let b_batch_total: usize = b_batch.iter().product::<usize>().max(1);

    let mut out_data = vec![0.0f32; batch_size * out_mat_stride];
    for batch_idx in 0..batch_size {
        let a_idx = if a_batch_total == 1 {
            0
        } else {
            batch_idx % a_batch_total
        };
        let b_idx = if b_batch_total == 1 {
            0
        } else {
            batch_idx % b_batch_total
        };
        let a_slice = &a_data[a_idx * a_mat_stride..(a_idx + 1) * a_mat_stride];
        let b_slice = &b_data[b_idx * b_mat_stride..(b_idx + 1) * b_mat_stride];
        let dst = &mut out_data[batch_idx * out_mat_stride..(batch_idx + 1) * out_mat_stride];
        yscv_kernels::matmul_2d_slices_trans_a(a_slice, m, k, b_slice, n, dst);
    }

    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    let out = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(matmul_node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_matmul(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    // Packed-INT4 fast path: weight is in the side-table populated by
    // `quantize_matmul_weights_int4_packed`. Compute via per-row GEMV
    // calls — `m_w = N` rows of length K each, output is the dot of
    // each row with the activation slice of length K.
    if let Some(packed) = env.packed_int4_weights.get(&node.inputs[1]).cloned() {
        let a = get_tensor(env, &node.name, &node.inputs[0])?;
        let a_shape = a.shape();
        let a_rank = a_shape.len();
        if a_rank >= 2 && a_shape[a_rank - 1] == packed.k {
            let m: usize = a_shape[..a_rank - 1].iter().product();
            let k = packed.k;
            let n = packed.m_w;
            let a_data = a.data();
            let mut out_data = vec![0.0_f32; m * n];
            // Threshold: GEMV per-row is fine for decode (M=1) and small
            // batches, but unpacks each weight nibble M times. Beyond a
            // handful of rows the GEMM kernel pays the per-tile unpack
            // once and reuses across rows. 8 is empirical break-even.
            const GEMM_THRESHOLD: usize = 8;
            if m >= GEMM_THRESHOLD {
                yscv_kernels::packed_int4_gemm_dispatch(
                    &packed.packed,
                    &packed.scales,
                    a_data,
                    &mut out_data,
                    m,
                    n,
                    k,
                    packed.group_size,
                );
            } else {
                for row in 0..m {
                    let act = &a_data[row * k..(row + 1) * k];
                    let dst = &mut out_data[row * n..(row + 1) * n];
                    yscv_kernels::packed_int4_gemv_dispatch(
                        &packed.packed,
                        &packed.scales,
                        act,
                        dst,
                        n,
                        k,
                        packed.group_size,
                    );
                }
            }
            let mut out_shape: Vec<usize> = a_shape[..a_rank - 1].to_vec();
            out_shape.push(n);
            let out =
                Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
            env.insert(node.outputs[0].clone(), out);
            return Ok(());
        }
    }

    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;

    if a.rank() <= 2 && b.rank() <= 2 {
        let out = matmul_2d(a, b).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), out);
        return Ok(());
    }

    // Batched matmul: [..., M, K] @ [..., K, N] → [..., M, N]
    // Delegates each 2D slice to matmul_2d (SIMD/blocked GEMM kernel).
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    let m = a_shape[a_rank - 2];
    let k = a_shape[a_rank - 1];
    let n = b_shape[b_rank - 1];

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];
    let max_batch_rank = a_batch.len().max(b_batch.len());
    let mut out_batch = Vec::with_capacity(max_batch_rank);
    for i in 0..max_batch_rank {
        let ad = if i < max_batch_rank - a_batch.len() {
            1
        } else {
            a_batch[i - (max_batch_rank - a_batch.len())]
        };
        let bd = if i < max_batch_rank - b_batch.len() {
            1
        } else {
            b_batch[i - (max_batch_rank - b_batch.len())]
        };
        out_batch.push(ad.max(bd));
    }

    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let a_mat_stride = m * k;
    let b_mat_stride = k * n;
    let out_mat_stride = m * n;
    let a_data = a.data();
    let b_data = b.data();
    let a_batch_total: usize = a_batch.iter().product::<usize>().max(1);
    let b_batch_total: usize = b_batch.iter().product::<usize>().max(1);

    let mut out_data = vec![0.0f32; batch_size * out_mat_stride];
    for batch_idx in 0..batch_size {
        let a_idx = if a_batch_total == 1 {
            0
        } else {
            batch_idx % a_batch_total
        };
        let b_idx = if b_batch_total == 1 {
            0
        } else {
            batch_idx % b_batch_total
        };
        let a_slice = &a_data[a_idx * a_mat_stride..(a_idx + 1) * a_mat_stride];
        let b_slice = &b_data[b_idx * b_mat_stride..(b_idx + 1) * b_mat_stride];
        let dst = &mut out_data[batch_idx * out_mat_stride..(batch_idx + 1) * out_mat_stride];

        // Zero-copy: call BLAS/GEMM directly on slices, no Tensor wrapping
        matmul_2d_slices(a_slice, m, k, b_slice, n, dst);
    }

    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    let out = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_einsum(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let equation = get_attr_string(node, "equation").unwrap_or_default();

    if equation == "ij,jk->ik" && node.inputs.len() == 2 {
        let a = get_tensor(env, &node.name, &node.inputs[0])?;
        let b = get_tensor(env, &node.name, &node.inputs[1])?;
        let result = yscv_kernels::matmul_2d(a, b).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    if equation == "ij->ji" && node.inputs.len() == 1 {
        let a = get_tensor(env, &node.name, &node.inputs[0])?;
        let result = a.transpose_2d().map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    Err(OnnxError::UnsupportedOpType {
        op_type: format!("Einsum({})", equation),
    })
}

pub(super) fn exec_qlinear_matmul(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let a_scale = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let a_zp = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];
    let b = get_tensor(env, &node.name, &node.inputs[3])?;
    let b_scale = get_tensor(env, &node.name, &node.inputs[4])?.data()[0];
    let b_zp = get_tensor(env, &node.name, &node.inputs[5])?.data()[0];
    let y_scale = get_tensor(env, &node.name, &node.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &node.name, &node.inputs[7])?.data()[0];

    // Symmetric fast path: both zero-points are 0, inputs are clean i8.
    // Pure dot-product via the SIMD kernel; final requantize folds the
    // composite scale `(a_scale * b_scale) / y_scale` and the output
    // zero-point. Works only on rank-2 (M,K)·(K,N) — higher-rank batched
    // matmuls fall through to the f32 reference below.
    if a_zp == 0.0
        && b_zp == 0.0
        && a.shape().len() == 2
        && b.shape().len() == 2
        && a.shape()[1] == b.shape()[0]
    {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let a_i8: Vec<i8> = a.data().iter().map(|&v| v as i8).collect();
        let b_i8: Vec<i8> = b.data().iter().map(|&v| v as i8).collect();
        let mut acc = vec![0_i32; m * n];
        yscv_kernels::int8_matmul_dispatch(&a_i8, &b_i8, m, k, n, &mut acc);
        let composite = (a_scale * b_scale) / y_scale;
        let quant: Vec<f32> = acc
            .iter()
            .map(|&v| ((v as f32) * composite + y_zp).round().clamp(-128.0, 127.0))
            .collect();
        let out = Tensor::from_vec(vec![m, n], quant).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), out);
        return Ok(());
    }

    let deq_a: Vec<f32> = a.data().iter().map(|&v| (v - a_zp) * a_scale).collect();
    let deq_b: Vec<f32> = b.data().iter().map(|&v| (v - b_zp) * b_scale).collect();

    let deq_a_t =
        Tensor::from_vec(a.shape().to_vec(), deq_a).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let deq_b_t =
        Tensor::from_vec(b.shape().to_vec(), deq_b).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let float_node = OnnxNode {
        name: node.name.clone(),
        op_type: "MatMul".to_string(),
        inputs: vec!["__qa".into(), "__qb_mat".into()],
        outputs: vec!["__qmm_out".into()],
        attributes: HashMap::new(),
    };
    env.insert("__qa".into(), deq_a_t);
    env.insert("__qb_mat".into(), deq_b_t);
    exec_matmul(&float_node, env)?;
    let float_out = env
        .remove("__qmm_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__qmm_out".into(),
        })?;
    env.remove("__qa");
    env.remove("__qb_mat");

    let quant: Vec<f32> = float_out
        .data()
        .iter()
        .map(|&v| (v / y_scale + y_zp).round().clamp(-128.0, 127.0))
        .collect();
    let out = Tensor::from_vec(float_out.shape().to_vec(), quant).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// Execute the ONNX GroupQueryAttention operator.
///
/// Inputs:
///   0: query  — `[batch, seq_q, num_q_heads * d_head]`
///   1: key    — `[batch, seq_k, num_kv_heads * d_head]`
///   2: value  — `[batch, seq_k, num_kv_heads * d_head]`
///
/// Attributes:
///   num_heads  (int) — number of query heads
///   kv_num_heads (int, optional) — number of KV heads (defaults to num_heads)
///
/// Output: `[batch, seq_q, num_q_heads * d_head]`
pub(super) fn exec_grouped_query_attention(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let q = get_tensor(env, &node.name, &node.inputs[0])?;
    let k = get_tensor(env, &node.name, &node.inputs[1])?;
    let v = get_tensor(env, &node.name, &node.inputs[2])?;

    let num_q_heads = get_attr_int(node, "num_heads").unwrap_or(1) as usize;
    let num_kv_heads = get_attr_int(node, "kv_num_heads")
        .map(|v| v as usize)
        .unwrap_or(num_q_heads);

    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    let batch = q_shape[0];
    let seq_q = q_shape[1];
    let seq_k = k_shape[1];
    let d_head = q_shape[2] / num_q_heads;
    let d_v = v_shape[2] / num_kv_heads;

    let groups = num_q_heads / num_kv_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    let mut out_data = vec![0.0f32; batch * seq_q * num_q_heads * d_head];

    for b in 0..batch {
        let q_batch_off = b * seq_q * num_q_heads * d_head;
        let k_batch_off = b * seq_k * num_kv_heads * d_head;
        let v_batch_off = b * seq_k * num_kv_heads * d_v;
        let o_batch_off = b * seq_q * num_q_heads * d_head;

        for qh in 0..num_q_heads {
            let kv_h = qh / groups;

            for sq in 0..seq_q {
                // Compute attention scores: dot(q, k^T) * scale
                let mut scores = Vec::with_capacity(seq_k);
                let mut max_score = f32::NEG_INFINITY;
                for sk in 0..seq_k {
                    let mut dot = 0.0f32;
                    for d in 0..d_head {
                        let qi = q_data[q_batch_off + sq * num_q_heads * d_head + qh * d_head + d];
                        let ki =
                            k_data[k_batch_off + sk * num_kv_heads * d_head + kv_h * d_head + d];
                        dot += qi * ki;
                    }
                    let s = dot * scale;
                    if s > max_score {
                        max_score = s;
                    }
                    scores.push(s);
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                if sum_exp > 0.0 {
                    let inv = 1.0 / sum_exp;
                    for s in &mut scores {
                        *s *= inv;
                    }
                }

                // Weighted sum of values
                for d in 0..d_head {
                    let mut acc = 0.0f32;
                    for sk in 0..seq_k {
                        let vi = v_data[v_batch_off + sk * num_kv_heads * d_v + kv_h * d_v + d];
                        acc += scores[sk] * vi;
                    }
                    out_data[o_batch_off + sq * num_q_heads * d_head + qh * d_head + d] = acc;
                }
            }
        }
    }

    let out =
        Tensor::from_vec(vec![batch, seq_q, num_q_heads * d_head], out_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_matmul_integer(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let a_zp = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        0.0
    };
    let b_zp = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        get_tensor(env, &node.name, &node.inputs[3])?.data()[0]
    } else {
        0.0
    };

    // Symmetric int8 fast path — no zero-point subtraction needed,
    // output is the raw int32 dot stored as f32 (MatMulInteger has no
    // requantize step). Rank-2 with matching K only.
    if a_zp == 0.0
        && b_zp == 0.0
        && a.shape().len() == 2
        && b.shape().len() == 2
        && a.shape()[1] == b.shape()[0]
    {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let a_i8: Vec<i8> = a.data().iter().map(|&v| v as i8).collect();
        let b_i8: Vec<i8> = b.data().iter().map(|&v| v as i8).collect();
        let mut acc = vec![0_i32; m * n];
        yscv_kernels::int8_matmul_dispatch(&a_i8, &b_i8, m, k, n, &mut acc);
        let out_data: Vec<f32> = acc.iter().map(|&v| v as f32).collect();
        let out = Tensor::from_vec(vec![m, n], out_data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), out);
        return Ok(());
    }

    let deq_a: Vec<f32> = a.data().iter().map(|&v| v - a_zp).collect();
    let deq_b: Vec<f32> = b.data().iter().map(|&v| v - b_zp).collect();

    let t_a = Tensor::from_vec(a.shape().to_vec(), deq_a).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    let t_b = Tensor::from_vec(b.shape().to_vec(), deq_b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let mm_node = OnnxNode {
        name: node.name.clone(),
        op_type: "MatMul".to_string(),
        inputs: vec!["__mmi_a".into(), "__mmi_b".into()],
        outputs: vec!["__mmi_out".into()],
        attributes: HashMap::new(),
    };
    env.insert("__mmi_a".into(), t_a);
    env.insert("__mmi_b".into(), t_b);
    exec_matmul(&mm_node, env)?;
    let out = env
        .remove("__mmi_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__mmi_out".into(),
        })?;
    env.remove("__mmi_a");
    env.remove("__mmi_b");

    let rounded: Vec<f32> = out.data().iter().map(|&v| v.round()).collect();
    let result =
        Tensor::from_vec(out.shape().to_vec(), rounded).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}
