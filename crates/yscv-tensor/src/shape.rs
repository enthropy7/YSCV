use super::tensor::{DimsVec, INLINE_CAP};

pub(crate) fn shape_element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
}

pub(crate) fn compute_strides(shape: &[usize]) -> Option<DimsVec> {
    if shape.len() <= INLINE_CAP {
        let mut buf = [0usize; INLINE_CAP];
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            buf[axis] = stride;
            stride = stride.checked_mul(shape[axis])?;
        }
        return Some(DimsVec::Inline {
            buf,
            len: shape.len() as u8,
        });
    }

    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride.checked_mul(shape[axis])?;
    }
    Some(DimsVec::Heap(strides))
}

pub(crate) fn broadcast_shape(left: &[usize], right: &[usize]) -> Option<Vec<usize>> {
    let max_rank = left.len().max(right.len());
    let left_pad = max_rank - left.len();
    let right_pad = max_rank - right.len();
    let mut out = Vec::with_capacity(max_rank);

    for axis in 0..max_rank {
        let ldim = if axis < left_pad {
            1
        } else {
            left[axis - left_pad]
        };
        let rdim = if axis < right_pad {
            1
        } else {
            right[axis - right_pad]
        };

        if ldim == rdim {
            out.push(ldim);
        } else if ldim == 1 {
            out.push(rdim);
        } else if rdim == 1 {
            out.push(ldim);
        } else {
            return None;
        }
    }
    Some(out)
}

pub(crate) fn broadcast_offset(shape: &[usize], strides: &[usize], out_coords: &[usize]) -> usize {
    if shape.is_empty() {
        return 0;
    }

    let axis_offset = out_coords.len() - shape.len();
    let mut out = 0usize;
    for axis in 0..shape.len() {
        let coord = if shape[axis] == 1 {
            0
        } else {
            out_coords[axis + axis_offset]
        };
        out += coord * strides[axis];
    }
    out
}

pub(crate) fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    if coords.is_empty() {
        return;
    }

    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < shape[axis] {
            return;
        }
        coords[axis] = 0;
    }
}
