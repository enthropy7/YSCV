#![cfg_attr(
    all(target_arch = "arm", feature = "neon-v7"),
    feature(stdarch_arm_neon_intrinsics, arm_target_feature)
)]
#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]

mod core;

pub use core::*;
