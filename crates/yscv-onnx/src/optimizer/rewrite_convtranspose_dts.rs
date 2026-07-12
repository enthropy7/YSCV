use yscv_tensor::Tensor;

use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

/// Rewrites `ConvTranspose(kernel=k×k, stride=k×k)` into the mathematically
/// identical pair `Conv 1×1 (k²·C_out channels)` + `DepthToSpace(blocksize=k,
/// mode=CRD)`.
///
/// Why: a k==s transposed convolution places, for every input pixel, an
/// independent k×k block of outputs — each output sub-position `(di, dj)` is
/// its own 1×1 convolution with weights `W[:, :, di, dj]`. Stacking those k²
/// convolutions channel-wise and unshuffling with DepthToSpace reproduces the
/// output exactly (same FLOPs), but routes through the GEMM-backed Conv path
/// instead of a scatter-style ConvTranspose kernel.
///
/// When it wins:
/// - always a *coverage* win for backends without a ConvTranspose kernel
///   (e.g. the MPSGraph backend);
/// - a throughput win where ConvTranspose is not GEMM-backed (current CPU
///   path: FastSAM-s proto upsample dropped ~20% end-to-end);
/// - near-neutral where a tuned ConvTranspose kernel exists — the extra
///   DepthToSpace is a single memory pass.
///
/// Applies only to the safe subset: square kernel with `k == stride`,
/// `group == 1`, zero pads/output_padding, weight present as an initializer.
/// Everything else is left untouched.
pub fn rewrite_convtranspose_dts(model: &mut OnnxModel) {
    let mut rewrites: Vec<(usize, Vec<OnnxNode>)> = Vec::new();

    for (idx, node) in model.nodes.iter().enumerate() {
        if node.op_type != "ConvTranspose" || node.inputs.len() < 2 {
            continue;
        }
        let ints = |name: &str| -> Option<Vec<i64>> {
            match node.attributes.get(name) {
                Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
                _ => None,
            }
        };
        let strides = ints("strides").unwrap_or_else(|| vec![1, 1]);
        let group = match node.attributes.get("group") {
            Some(OnnxAttribute::Int(g)) => *g,
            _ => 1,
        };
        let pads = ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let out_pad = ints("output_padding").unwrap_or_else(|| vec![0, 0]);
        let dilations = ints("dilations").unwrap_or_else(|| vec![1, 1]);

        let Some(weight) = model.initializers.get(&node.inputs[1]) else {
            continue;
        };
        let w_shape = weight.shape().to_vec();
        if w_shape.len() != 4 {
            continue;
        }
        let (c_in, c_out, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let k = kh;
        let safe = kh == kw
            && strides.len() == 2
            && strides.iter().all(|&s| s == k as i64)
            && group == 1
            && pads.iter().all(|&p| p == 0)
            && out_pad.iter().all(|&p| p == 0)
            && dilations.iter().all(|&d| d == 1);
        if !safe {
            continue;
        }

        // W[C_in, C_out, k, k] -> Conv weight [k²·C_out, C_in, 1, 1],
        // выход канала (co·k² + di·k + dj) = вклад суб-позиции (di, dj).
        let w = weight.data();
        let mut wc = vec![0.0f32; c_out * k * k * c_in];
        for ci in 0..c_in {
            for co in 0..c_out {
                for di in 0..k {
                    for dj in 0..k {
                        let src = ((ci * c_out + co) * k + di) * k + dj;
                        let dst_ch = (co * k + di) * k + dj;
                        wc[dst_ch * c_in + ci] = w[src];
                    }
                }
            }
        }
        let base = if node.name.is_empty() {
            node.outputs[0].clone()
        } else {
            node.name.clone()
        };
        let w_name = format!("{base}_dts_w");
        let Ok(w_tensor) = Tensor::from_vec(vec![c_out * k * k, c_in, 1, 1], wc) else {
            continue;
        };

        let mut conv_inputs = vec![node.inputs[0].clone(), w_name.clone()];
        let mut bias_tensor: Option<(String, Tensor)> = None;
        if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
            let Some(bias) = model.initializers.get(&node.inputs[2]) else {
                continue;
            };
            let b = bias.data();
            if b.len() != c_out {
                continue;
            }
            let mut bb = vec![0.0f32; c_out * k * k];
            for co in 0..c_out {
                for kk in 0..k * k {
                    bb[co * k * k + kk] = b[co];
                }
            }
            let b_name = format!("{base}_dts_b");
            let Ok(bt) = Tensor::from_vec(vec![c_out * k * k], bb) else {
                continue;
            };
            conv_inputs.push(b_name.clone());
            bias_tensor = Some((b_name, bt));
        }

        let mid = format!("{base}_dts_conv");
        let conv = OnnxNode {
            op_type: "Conv".to_owned(),
            name: format!("{base}_c1x1"),
            inputs: conv_inputs,
            outputs: vec![mid.clone()],
            attributes: [
                ("strides".to_owned(), OnnxAttribute::Ints(vec![1, 1])),
                ("pads".to_owned(), OnnxAttribute::Ints(vec![0, 0, 0, 0])),
                ("kernel_shape".to_owned(), OnnxAttribute::Ints(vec![1, 1])),
            ]
            .into_iter()
            .collect(),
        };
        let dts = OnnxNode {
            op_type: "DepthToSpace".to_owned(),
            name: format!("{base}_dts"),
            inputs: vec![mid],
            outputs: node.outputs.clone(),
            attributes: [
                ("blocksize".to_owned(), OnnxAttribute::Int(k as i64)),
                ("mode".to_owned(), OnnxAttribute::String("CRD".to_owned())),
            ]
            .into_iter()
            .collect(),
        };

        model.initializers.insert(w_name, w_tensor);
        if let Some((b_name, bt)) = bias_tensor {
            model.initializers.insert(b_name, bt);
        }
        rewrites.push((idx, vec![conv, dts]));
    }

    // заменяем узлы с конца, чтобы индексы не съехали
    for (idx, replacement) in rewrites.into_iter().rev() {
        model.nodes.splice(idx..=idx, replacement);
    }
}
