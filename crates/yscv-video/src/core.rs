#[path = "audio.rs"]
pub mod audio;
#[path = "av1_decoder.rs"]
mod av1_decoder;
#[path = "av1_obu.rs"]
pub mod av1_obu;
#[path = "camera.rs"]
mod camera;
#[path = "cavlc.rs"]
pub mod cavlc;
#[path = "codec.rs"]
mod codec;
#[path = "convert.rs"]
mod convert;
#[cfg(feature = "drm")]
#[path = "drm_output.rs"]
pub mod drm_output;
#[path = "error.rs"]
mod error;
#[path = "frame.rs"]
mod frame;
#[path = "frame_pipeline.rs"]
#[allow(unsafe_code)]
pub mod frame_pipeline;
#[path = "frame_pipeline_5stage.rs"]
#[allow(unsafe_code)]
pub mod frame_pipeline_5stage;
#[path = "latency_histogram.rs"]
pub mod latency_histogram;
#[path = "frame_common.rs"]
pub mod frame_common;
#[cfg(target_os = "linux")]
#[path = "framebuffer.rs"]
#[allow(unsafe_code)]
pub mod framebuffer;
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
#[path = "h264_encoder.rs"]
#[allow(unsafe_code)]
pub mod h264_encoder;
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
#[path = "hevc_parallel.rs"]
#[allow(unsafe_code)]
mod hevc_parallel;
#[path = "hevc_params.rs"]
pub mod hevc_params;
#[path = "hevc_syntax.rs"]
pub mod hevc_syntax;
#[path = "hevc_transform.rs"]
pub mod hevc_transform;
#[path = "hw_decode.rs"]
pub mod hw_decode;
#[path = "mavlink.rs"]
pub mod mavlink;
#[path = "mjpeg.rs"]
mod mjpeg;
#[path = "mkv.rs"]
pub mod mkv;
#[path = "overlay.rs"]
pub mod overlay;
#[path = "realtime.rs"]
#[allow(unsafe_code)]
pub mod realtime;
#[cfg(feature = "rga")]
#[path = "rga.rs"]
pub mod rga;
#[path = "source.rs"]
mod source;
#[path = "stream.rs"]
mod stream;
#[cfg(target_os = "linux")]
#[path = "v4l2.rs"]
#[allow(unsafe_code)]
pub mod v4l2;
#[path = "video_io.rs"]
mod video_io;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "proptest_tests.rs"]
mod proptest_tests;

pub const CRATE_ID: &str = "yscv-video";

pub use av1_decoder::{
    Av1Decoder, Av1IntraMode, av1_intra_dc, av1_intra_h, av1_intra_paeth, av1_intra_smooth,
    av1_intra_v, av1_inverse_adst_4x4, av1_inverse_dct_4x4,
};
pub use av1_obu::{
    Av1FrameHeader, Av1ObuType, Av1SequenceHeader, parse_frame_header, parse_obus,
    parse_sequence_header,
};
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
pub use frame_pipeline::{
    FramePipeline, PipelineBBox, PipelineDetection, PipelineStats, SlotMut, SlotRef, run_pipeline,
};
#[cfg(target_os = "linux")]
pub use framebuffer::LinuxFramebuffer;
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
pub use h264_encoder::{H264Encoder, forward_dct_4x4, rgb8_to_yuv420};
pub use h264_motion::{MotionVector, ReferenceFrameBuffer, motion_compensate_16x16, predict_mv};
pub use h264_params::{
    Pps, SliceHeader, Sps, parse_pps, parse_sps, remove_emulation_prevention_with_mapping,
};
pub use h264_transform::{dequant_4x4, dequant_8x8, inverse_dct_4x4, inverse_dct_8x8};
pub use h264_yuv::{
    nv12_to_rgb8, yuv_to_rgb8_generic, yuv420_p16_to_rgb16, yuv420_to_rgb8, yuv420_to_rgb8_into,
    yuyv_to_rgb8,
};
pub use hevc_cabac::{CabacDecoder, ContextModel};
pub use hevc_decoder::HevcNalUnitType;
pub use hevc_decoder::{
    CodingTreeUnit, DecodedCu, HevcDecoder, HevcIntraMode, HevcPps, HevcPredMode, HevcSliceHeader,
    HevcSliceType, HevcSps, HevcTileRect, HevcVps, decode_coding_tree, hevc_dequant,
    hevc_frame_dimensions, hevc_inverse_dct_4x4, hevc_inverse_dct_8x8, hevc_inverse_dct_16x16,
    hevc_inverse_dct_32x32, hevc_inverse_dst_4x4, intra_predict_angular, intra_predict_dc,
    intra_predict_planar, parse_hevc_pps, parse_hevc_slice_header_full, parse_hevc_sps,
    parse_hevc_vps, pps_tile_rects,
};
pub use hevc_filter::{
    BETA_TABLE as HEVC_BETA_TABLE, HEVC_CHROMA_FILTER, SaoParams, SaoType,
    TC_TABLE as HEVC_TC_TABLE, chroma_interpolate_row, chroma_interpolate_sample,
    derive_beta as hevc_derive_beta, derive_chroma_qp, derive_tc as hevc_derive_tc,
    finalize_hevc_frame, hevc_apply_sao, hevc_apply_sao_chroma, hevc_boundary_strength,
    hevc_deblock_frame, hevc_filter_edge_chroma, hevc_filter_edge_luma, parse_sao_params,
    reconstruct_chroma_plane,
};
pub use hevc_inter::{
    HevcDpb, HevcMv, HevcMvField, HevcReferencePicture, build_amvp_candidates,
    build_merge_candidates, hevc_bipred_average, hevc_mc_luma, hevc_unipred_clip,
    parse_inter_prediction, parse_merge_idx, parse_mvd as hevc_parse_mvd,
};
pub use hevc_syntax::{
    CodingUnitData, HevcSliceCabacState, NUM_CABAC_CONTEXTS, TileBounds, build_mpm_list,
    decode_coding_tree_cabac, parse_coding_unit, parse_split_cu_flag, parse_transform_unit,
};
#[cfg(target_os = "linux")]
pub use mavlink::MavlinkSerial;
pub use mavlink::{
    MavlinkHeader, MavlinkMessage, MavlinkParser, TelemetryUpdate, apply_telemetry_update,
    parse_mavlink_frame, telemetry_from_mavlink,
};
pub use mjpeg::decode_mjpeg_to_rgb8;
pub use overlay::{
    SharedTelemetry, TelemetryData, draw_rect as overlay_draw_rect, draw_text as overlay_draw_text,
    overlay_detections, overlay_telemetry,
};
pub use source::{FrameSource, InMemoryFrameSource};
pub use stream::FrameStream;
#[cfg(target_os = "linux")]
pub use v4l2::{V4l2Camera, V4l2DmaBufGuard, V4l2PixelFormat};
pub use video_io::{
    ImageSequenceReader, Mp4VideoReader, RawVideoReader, RawVideoWriter, VideoMeta,
};
