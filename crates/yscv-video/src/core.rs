#[path = "audio.rs"]
pub mod audio;
#[path = "camera.rs"]
mod camera;
#[path = "cavlc.rs"]
pub mod cavlc;
#[path = "codec.rs"]
mod codec;
#[path = "convert.rs"]
mod convert;
#[path = "error.rs"]
mod error;
#[path = "frame.rs"]
mod frame;
#[path = "h264_bitstream.rs"]
pub mod h264_bitstream;
#[path = "h264_bslice.rs"]
pub mod h264_bslice;
#[path = "h264_cabac.rs"]
pub mod h264_cabac;
#[path = "h264_deblock.rs"]
pub mod h264_deblock;
#[path = "h264_decoder.rs"]
mod h264_decoder;
#[path = "h264_motion.rs"]
pub mod h264_motion;
#[path = "h264_params.rs"]
pub mod h264_params;
#[path = "h264_transform.rs"]
pub mod h264_transform;
#[path = "h264_yuv.rs"]
pub mod h264_yuv;
#[path = "hevc_cabac.rs"]
pub mod hevc_cabac;
#[path = "hevc_decoder.rs"]
pub mod hevc_decoder;
#[path = "hevc_filter.rs"]
pub mod hevc_filter;
#[path = "hevc_inter.rs"]
pub mod hevc_inter;
#[path = "hevc_params.rs"]
pub mod hevc_params;
#[path = "hevc_syntax.rs"]
pub mod hevc_syntax;
#[path = "hevc_transform.rs"]
pub mod hevc_transform;
#[path = "hw_decode.rs"]
pub mod hw_decode;
#[path = "mkv.rs"]
pub mod mkv;
#[path = "source.rs"]
mod source;
#[path = "stream.rs"]
mod stream;
#[path = "video_io.rs"]
mod video_io;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

pub const CRATE_ID: &str = "yscv-video";

pub use camera::{
    CameraConfig, CameraDeviceInfo, CameraFrameSource, filter_camera_devices, list_camera_devices,
    query_camera_devices, resolve_camera_device, resolve_camera_device_index,
};
pub use codec::{
    DecodedFrame, EncodedPacket, Mp4Box, NalUnit, NalUnitType, VideoCodec, VideoDecoder,
    VideoEncoder, extract_parameter_sets, find_box, parse_annex_b, parse_child_boxes,
    parse_mp4_boxes,
};
pub use convert::normalize_rgb8_to_f32_inplace;
pub use error::VideoError;
pub use frame::{Frame, PixelFormat, Rgb8Frame};
pub use h264_bitstream::BitstreamReader;
pub use h264_bslice::{BPredMode, BiMotionVector, decode_b_macroblock, motion_compensate_bipred};
pub use h264_cabac::{
    CabacContext, CabacDecoder as H264CabacDecoder, EntropyCodingMode,
    NUM_CABAC_CONTEXTS as H264_NUM_CABAC_CONTEXTS, decode_coded_block_flag,
    decode_exp_golomb_bypass, decode_fixed_length, decode_mb_type_i_slice, decode_mb_type_p_slice,
    decode_residual_block_cabac, decode_truncated_unary, decode_unary, init_cabac_contexts,
};
pub use h264_deblock::{compute_boundary_strength, deblock_edge_luma, deblock_frame};
pub use h264_decoder::H264Decoder;
pub use h264_motion::{MotionVector, ReferenceFrameBuffer, motion_compensate_16x16, predict_mv};
pub use h264_params::{Pps, SliceHeader, Sps, parse_pps, parse_sps};
pub use h264_transform::{dequant_4x4, dequant_8x8, inverse_dct_4x4, inverse_dct_8x8};
pub use h264_yuv::yuv420_to_rgb8;
pub use hevc_cabac::{CabacDecoder, ContextModel};
pub use hevc_decoder::HevcNalUnitType;
pub use hevc_decoder::{
    CodingTreeUnit, DecodedCu, HevcDecoder, HevcIntraMode, HevcPps, HevcPredMode, HevcSliceHeader,
    HevcSliceType, HevcSps, HevcVps, decode_coding_tree, hevc_dequant, hevc_frame_dimensions,
    hevc_inverse_dct_4x4, hevc_inverse_dct_8x8, hevc_inverse_dct_16x16, hevc_inverse_dct_32x32,
    hevc_inverse_dst_4x4, intra_predict_angular, intra_predict_dc, intra_predict_planar,
    parse_hevc_pps, parse_hevc_sps, parse_hevc_vps,
};
pub use hevc_filter::{
    BETA_TABLE as HEVC_BETA_TABLE, HEVC_CHROMA_FILTER, SaoParams, SaoType,
    TC_TABLE as HEVC_TC_TABLE, chroma_interpolate_row, chroma_interpolate_sample,
    derive_beta as hevc_derive_beta, derive_chroma_qp, derive_tc as hevc_derive_tc,
    finalize_hevc_frame, hevc_apply_sao, hevc_boundary_strength, hevc_deblock_frame,
    hevc_filter_edge_chroma, hevc_filter_edge_luma, parse_sao_params, reconstruct_chroma_plane,
};
pub use hevc_inter::{
    HevcDpb, HevcMv, HevcMvField, HevcReferencePicture, build_amvp_candidates,
    build_merge_candidates, hevc_bipred_average, hevc_mc_luma, hevc_unipred_clip,
    parse_inter_prediction, parse_merge_idx, parse_mvd as hevc_parse_mvd,
};
pub use hevc_syntax::{
    CodingUnitData, HevcSliceCabacState, NUM_CABAC_CONTEXTS, build_mpm_list,
    decode_coding_tree_cabac, parse_coding_unit, parse_split_cu_flag, parse_transform_unit,
};
pub use source::{FrameSource, InMemoryFrameSource};
pub use stream::FrameStream;
pub use video_io::{
    ImageSequenceReader, Mp4VideoReader, RawVideoReader, RawVideoWriter, VideoMeta,
};
