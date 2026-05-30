//! Shape manipulation: transpose/permute/squeeze/cat/stack/select/narrow.

use super::super::aligned::AlignedVec;
use super::super::error::TensorError;
use super::super::shape::{compute_strides, increment_coords, shape_element_count};
use super::super::tensor::Tensor;

impl Tensor {
    // ── Shape manipulation ──────────────────────────────────────────────

    /// 2D matrix transpose. Requires rank-2 input.
    ///
    /// # Safety
    /// `AlignedVec::uninitialized` allocates without zeroing. The tiled loop
    /// writes every element before anything reads from the buffer.
    #[allow(unsafe_code)]
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::InvalidAxis {
                axis: 1,
                rank: self.rank(),
            });
        }
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        // SAFETY: every element is written by the tiled loop below before we read.
        let mut out_data = AlignedVec::<f32>::uninitialized(rows * cols);
        let src = self.data();

        // Tiled transpose with 8x8 blocks for cache efficiency.
        const TILE: usize = 8;
        let rr = rows / TILE * TILE;
        let cc = cols / TILE * TILE;

        for ii in (0..rr).step_by(TILE) {
            for jj in (0..cc).step_by(TILE) {
                for r in ii..ii + TILE {
                    for c in jj..jj + TILE {
                        out_data[c * rows + r] = src[r * cols + c];
                    }
                }
            }
            // Right edge (columns beyond cc)
            for r in ii..ii + TILE {
                for c in cc..cols {
                    out_data[c * rows + r] = src[r * cols + c];
                }
            }
        }
        // Bottom edge (rows beyond rr)
        for r in rr..rows {
            for c in 0..cols {
                out_data[c * rows + r] = src[r * cols + c];
            }
        }

        Tensor::from_aligned(vec![cols, rows], out_data)
    }

    /// General axis permutation (like NumPy `transpose`/`permute`).
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                got: axes.len(),
            });
        }
        let rank = self.rank();
        let mut seen = vec![false; rank];
        for &a in axes {
            if a >= rank {
                return Err(TensorError::InvalidAxis { axis: a, rank });
            }
            seen[a] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(TensorError::InvalidAxis { axis: 0, rank });
        }

        let src_shape = self.shape();
        let mut out_shape = vec![0usize; rank];
        for (dst, &src_axis) in axes.iter().enumerate() {
            out_shape[dst] = src_shape[src_axis];
        }
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;

        // ── Fast path: tiled 2D transpose for common 4D permutations ──
        // Uses unsafe pointer arithmetic to eliminate bounds checks in hot inner loops.
        // NHWC→NCHW [0,3,1,2]: transpose inner [H*W, C] → [C, H*W]
        if rank == 4 && axes == [0, 3, 1, 2] {
            let (n, h, w, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let hw = h * w;
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..n {
                    let s_base = src_ptr.add(batch * hw * c);
                    let d_base = dst_ptr.add(batch * c * hw);
                    for i0 in (0..hw).step_by(TILE) {
                        let ie = (i0 + TILE).min(hw);
                        for j0 in (0..c).step_by(TILE) {
                            let je = (j0 + TILE).min(c);
                            for i in i0..ie {
                                let s_row = s_base.add(i * c);
                                for j in j0..je {
                                    *d_base.add(j * hw + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // NCHW→NHWC [0,2,3,1]: transpose inner [C, H*W] → [H*W, C]
        if rank == 4 && axes == [0, 2, 3, 1] {
            let (n, c, h, w) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let hw = h * w;
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..n {
                    let s_base = src_ptr.add(batch * c * hw);
                    let d_base = dst_ptr.add(batch * hw * c);
                    for i0 in (0..c).step_by(TILE) {
                        let ie = (i0 + TILE).min(c);
                        for j0 in (0..hw).step_by(TILE) {
                            let je = (j0 + TILE).min(hw);
                            for i in i0..ie {
                                let s_row = s_base.add(i * hw);
                                for j in j0..je {
                                    *d_base.add(j * c + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // 3D swap last two dims [0,2,1]: transpose [A, B, C] → [A, C, B]
        if rank == 3 && axes == [0, 2, 1] {
            let (a, b, c) = (src_shape[0], src_shape[1], src_shape[2]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..a {
                    let s_base = src_ptr.add(batch * b * c);
                    let d_base = dst_ptr.add(batch * c * b);
                    for i0 in (0..b).step_by(TILE) {
                        let ie = (i0 + TILE).min(b);
                        for j0 in (0..c).step_by(TILE) {
                            let je = (j0 + TILE).min(c);
                            for i in i0..ie {
                                let s_row = s_base.add(i * c);
                                for j in j0..je {
                                    *d_base.add(j * b + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }

        // [0,1,3,2]: swap last two dims in 4D → [N, A, C, B]
        // For each (n, a), tiled 2D transpose of [B, C] → [C, B].
        if rank == 4 && axes == [0, 1, 3, 2] {
            let (nn, a, b, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    for aa in 0..a {
                        let base = (n * a + aa) * b * c;
                        let s_base = src_ptr.add(base);
                        let d_base = dst_ptr.add(base); // same offset, different shape
                        for i0 in (0..b).step_by(TILE) {
                            let ie = (i0 + TILE).min(b);
                            for j0 in (0..c).step_by(TILE) {
                                let je = (j0 + TILE).min(c);
                                for i in i0..ie {
                                    let s_row = s_base.add(i * c);
                                    for j in j0..je {
                                        *d_base.add(j * b + i) = *s_row.add(j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // [0,2,1,3]: swap dims 1↔2 in 4D → [N, B, A, C]
        // Each element in the swap is a contiguous block of C floats — use memcpy.
        if rank == 4 && axes == [0, 2, 1, 3] {
            let (nn, a, b, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    let s_batch = src_ptr.add(n * a * b * c);
                    let d_batch = dst_ptr.add(n * b * a * c);
                    for aa in 0..a {
                        for bb in 0..b {
                            std::ptr::copy_nonoverlapping(
                                s_batch.add(aa * b * c + bb * c),
                                d_batch.add(bb * a * c + aa * c),
                                c,
                            );
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // [0,3,2,1]: swap dims 1↔3 in 4D → [N, D, B, A]
        // For each (n, b), tiled 2D transpose of [A, D] → [D, A] with strides.
        if rank == 4 && axes == [0, 3, 2, 1] {
            let (nn, a, b, d) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            let src_a_stride = b * d;
            let dst_d_stride = b * a;
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    for bb in 0..b {
                        let s_base = src_ptr.add(n * a * b * d + bb * d);
                        let d_base = dst_ptr.add(n * d * b * a + bb * a);
                        for i0 in (0..a).step_by(TILE) {
                            let ie = (i0 + TILE).min(a);
                            for j0 in (0..d).step_by(TILE) {
                                let je = (j0 + TILE).min(d);
                                for i in i0..ie {
                                    for j in j0..je {
                                        *d_base.add(j * dst_d_stride + i) =
                                            *s_base.add(i * src_a_stride + j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // 2D transpose [1,0]: swap rows and cols
        if rank == 2 && axes == [1, 0] {
            let (rows, cols) = (src_shape[0], src_shape[1]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for i0 in (0..rows).step_by(TILE) {
                    let ie = (i0 + TILE).min(rows);
                    for j0 in (0..cols).step_by(TILE) {
                        let je = (j0 + TILE).min(cols);
                        for i in i0..ie {
                            let s_row = src_ptr.add(i * cols);
                            for j in j0..je {
                                *dst_ptr.add(j * rows + i) = *s_row.add(j);
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }

        // ── General fallback: coordinate-based scatter ──
        let out_strides = compute_strides(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: out_shape.clone(),
        })?;
        let mut out_data = vec![0.0f32; out_count];

        let mut in_coords = vec![0usize; rank];
        for &val in self.data().iter() {
            let mut out_offset = 0usize;
            for (dst_axis, &src_axis) in axes.iter().enumerate() {
                out_offset += in_coords[src_axis] * out_strides[dst_axis];
            }
            out_data[out_offset] = val;
            increment_coords(&mut in_coords, src_shape);
        }

        Tensor::from_vec(out_shape, out_data)
    }

    /// Insert a length-1 dimension at the given axis.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, TensorError> {
        if axis > self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank() + 1,
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        self.reshape(new_shape)
    }

    /// Remove a length-1 dimension at the given axis.
    pub fn squeeze(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if self.shape()[axis] != 1 {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.remove(axis);
        self.reshape(new_shape)
    }

    /// Concatenate tensors along an axis. All tensors must have the same
    /// shape except along the concatenation axis.
    pub fn cat(tensors: &[&Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::SizeMismatch {
                shape: vec![],
                data_len: 0,
            });
        }
        let rank = tensors[0].rank();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }
        for t in &tensors[1..] {
            if t.rank() != rank {
                return Err(TensorError::ShapeMismatch {
                    left: tensors[0].shape().to_vec(),
                    right: t.shape().to_vec(),
                });
            }
            for (a, (&d0, &di)) in tensors[0].shape().iter().zip(t.shape().iter()).enumerate() {
                if a != axis && d0 != di {
                    return Err(TensorError::ShapeMismatch {
                        left: tensors[0].shape().to_vec(),
                        right: t.shape().to_vec(),
                    });
                }
            }
        }

        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;

        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = out_shape[axis + 1..].iter().product();

        // Write directly into AlignedVec to avoid the double-copy through
        // Vec -> AlignedVec::from_vec.
        let mut out_data = AlignedVec::<f32>::uninitialized(out_count);

        if inner == 1 && tensors.len() <= 8 {
            // Last-axis concat: write entire output in outer-major order.
            let axis_lens: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
            let dst = out_data.as_mut_slice();
            let mut dst_off = 0;
            for o in 0..outer {
                for (ti, t) in tensors.iter().enumerate() {
                    let al = axis_lens[ti];
                    let src_off = o * al;
                    dst[dst_off..dst_off + al].copy_from_slice(&t.data()[src_off..src_off + al]);
                    dst_off += al;
                }
            }
        } else {
            let dst = out_data.as_mut_slice();
            let mut dst_off = 0;
            for o in 0..outer {
                for t in tensors {
                    let t_axis_len = t.shape()[axis];
                    let chunk_len = t_axis_len * inner;
                    let chunk_start = o * chunk_len;
                    dst[dst_off..dst_off + chunk_len]
                        .copy_from_slice(&t.data()[chunk_start..chunk_start + chunk_len]);
                    dst_off += chunk_len;
                }
            }
        }

        Tensor::from_aligned(out_shape, out_data)
    }

    /// Stack tensors along a new axis. All tensors must have identical shapes.
    pub fn stack(tensors: &[&Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::SizeMismatch {
                shape: vec![],
                data_len: 0,
            });
        }
        if axis > tensors[0].rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: tensors[0].rank() + 1,
            });
        }
        let expanded: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(axis))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&Self> = expanded.iter().collect();
        Self::cat(&refs, axis)
    }

    /// Select a single slice along an axis, removing that axis from the output.
    pub fn select(&self, axis: usize, index: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if index >= self.shape()[axis] {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index,
                dim: self.shape()[axis],
            });
        }
        let outer: usize = self.shape()[..axis].iter().product();
        let axis_len = self.shape()[axis];
        let inner: usize = self.shape()[axis + 1..].iter().product();

        let mut out_data = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            let base = o * axis_len * inner + index * inner;
            out_data.extend_from_slice(&self.data()[base..base + inner]);
        }

        let mut out_shape = self.shape().to_vec();
        out_shape.remove(axis);
        Tensor::from_vec(out_shape, out_data)
    }

    /// Narrow (slice) along an axis: extract elements `start..start+length`.
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if start + length > self.shape()[axis] {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: start + length,
                dim: self.shape()[axis],
            });
        }
        let outer: usize = self.shape()[..axis].iter().product();
        let axis_len = self.shape()[axis];
        let inner: usize = self.shape()[axis + 1..].iter().product();

        let mut out_data = Vec::with_capacity(outer * length * inner);
        for o in 0..outer {
            let base = o * axis_len * inner + start * inner;
            out_data.extend_from_slice(&self.data()[base..base + length * inner]);
        }

        let mut out_shape = self.shape().to_vec();
        out_shape[axis] = length;
        Tensor::from_vec(out_shape, out_data)
    }
}
