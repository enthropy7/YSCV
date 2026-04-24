// ===========================================================================
// SIMD-accelerated element-wise operations
// ===========================================================================
//
// Sub-modules grouped by operation type:
//   - exp:        fast-exp helpers (SSE/AVX/NEON), exp_slice, sub_exp, tanh
//   - activation: ReLU, Sigmoid, SiLU (dispatch + inplace + all SIMD impls)
//   - reduce:     max_reduce, add_reduce dispatchers + impls
//   - binary:     binary_same_shape_dispatch + impls, mul_scalar_inplace + impls
//   - fma:        fma_slice_dispatch + impls, matmul_row_dispatch + impls
//   - softmax:    softmax_row_fused_dispatch + impls, log_softmax_row_fused_dispatch + impls

mod activation;
mod binary;
pub(crate) mod exp;
mod fma;
mod reduce;
mod softmax;

#[cfg(test)]
mod tests;

// Re-export everything so callers see a flat namespace (same as before the split).

// From activation
pub use activation::{
    bias_add_nhwc_dispatch, bias_relu_nhwc_dispatch, bias_silu_nhwc_dispatch,
    fused_row_epilogue_dispatch, relu_slice_dispatch, relu_to_slice_dispatch,
    sigmoid_slice_dispatch, silu_inplace, silu_slice_dispatch,
};
#[allow(unused_imports)]
pub(crate) use activation::{sigmoid_scalar, sigmoid_slice};

// From exp
pub use exp::{exp_slice_dispatch, sub_exp_slice_dispatch, tanh_slice_dispatch};

// From reduce
pub use reduce::{add_reduce_dispatch, max_reduce_dispatch};

// From binary
#[allow(unused_imports)]
pub use binary::{
    add_inplace_dispatch, add_relu_inplace_dispatch, binary_same_shape_dispatch,
    mul_scalar_inplace_dispatch,
};

// From fma
pub use fma::{fma_slice_dispatch, matmul_row_dispatch, matmul_row_set_dispatch};

// From softmax
pub use softmax::{log_softmax_row_fused_dispatch, softmax_row_fused_dispatch};
