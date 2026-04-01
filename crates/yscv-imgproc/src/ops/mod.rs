/// Wrapper to send raw mutable pointers across thread boundaries.
/// SAFETY: callers must ensure non-overlapping access from each thread.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    pub(crate) fn ptr(self) -> *mut T {
        self.0
    }
}

/// Wrapper to send raw const pointers across thread boundaries.
/// SAFETY: callers must ensure the pointed-to data lives long enough.
#[derive(Clone, Copy)]
pub(crate) struct SendConstPtr<T>(pub(crate) *const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> SendConstPtr<T> {
    #[inline(always)]
    pub(crate) fn ptr(self) -> *const T {
        self.0
    }
}

mod augment;
mod bitwise;
mod brief;
mod calibration;
mod color;
mod contours;
mod draw;
mod f32_ops;
mod fast;
mod features;
mod filter;
mod flow;
mod geometry;
mod histogram;
mod inpaint;
mod intensity;
mod io;
mod morphology;
mod nms;
mod normalize;
mod orb;
mod preprocess;
mod resize;
mod stereo;
mod surf;
mod threshold;
mod u8_canny;
mod u8_color;
mod u8_features;
mod u8_filters;
mod u8_harris;
mod u8_resize;
mod u8ops;

pub use augment::{
    color_jitter, elastic_transform, random_crop, random_erasing, random_horizontal_flip,
    random_rotation, random_vertical_flip,
};
pub use bitwise::{add_weighted, bitwise_and, bitwise_not, bitwise_or, in_range};
pub use brief::{BriefDescriptor, compute_brief, hamming_distance};
pub use calibration::{CameraIntrinsics, project_points, triangulate_points, undistort_points};
pub use color::{
    hsv_to_rgb, lab_to_rgb, rgb_to_bgr, rgb_to_grayscale, rgb_to_hsv, rgb_to_lab, rgb_to_yuv,
    yuv_to_rgb,
};
pub use contours::{
    ComponentStats, Contour, RegionProp, approx_poly_dp, arc_length, bounding_rect,
    connected_components_with_stats, contour_area, convex_hull, find_contours, fit_ellipse,
    homography_4pt, hu_moments, min_area_rect, ransac_homography, region_props,
};
pub use draw::{
    Detection as DrawDetection, draw_circle, draw_detections, draw_line, draw_polylines, draw_rect,
    draw_text, draw_text_scaled, fill_poly,
};
pub use f32_ops::{
    box_blur_3x3_f32, dilate_3x3_f32, gaussian_blur_3x3_f32, grayscale_f32, sobel_3x3_f32,
    threshold_binary_f32,
};
pub use fast::{Keypoint, fast9_detect};
pub use features::{
    HarrisKeypoint, HoughLine, OrbDescriptor, blob_log, compute_gradient_orientation,
    corner_sub_pix, distance_transform, fast_corners, gaussian_pyramid, good_features_to_track,
    harris_corners, hog_cell_descriptor, hough_circles, hough_lines, orb_descriptors,
    orb_hamming_distance, orb_match, sift_descriptor, sift_match,
};
pub use filter::{
    bilateral_filter, box_blur_3x3, filter2d, gaussian_blur_3x3, gaussian_blur_5x5, laplacian_3x3,
    median_blur_3x3, median_filter,
};
pub use flow::{
    FarnebackConfig, dense_optical_flow, farneback_flow, lucas_kanade_optical_flow, watershed,
};
pub use geometry::{
    crop, flip_horizontal, flip_vertical, pad_constant, rotate90_cw, scharr_3x3_gradients,
    scharr_3x3_magnitude, sobel_3x3_gradients, sobel_3x3_magnitude, warp_affine, warp_perspective,
};
pub use histogram::{clahe, histogram_256, histogram_equalize, integral_image};
pub use inpaint::inpaint_telea;
pub use intensity::{adjust_gamma, adjust_log, rescale_intensity};
pub use io::{imread, imread_gray, imwrite};
pub use morphology::{
    closing_3x3, dilate, dilate_3x3, erode, erode_3x3, morph_blackhat, morph_gradient_3x3,
    morph_tophat, opening_3x3, remove_small_objects, skeletonize,
};
pub use nms::{BBox, TemplateMatchMethod, TemplateMatchResult, nms, template_match};
pub use normalize::normalize;
pub use orb::{OrbConfig, OrbFeature, detect_orb, match_features};
pub use preprocess::{center_crop, chw_to_hwc, hwc_to_chw, imagenet_preprocess, normalize_image};
pub use resize::{resize_bilinear, resize_nearest};
pub use stereo::{StereoConfig, stereo_block_matching};
pub use surf::{
    SurfDescriptor, SurfKeypoint, build_integral_image, compute_surf_descriptors,
    detect_surf_keypoints, match_surf_descriptors,
};
pub use threshold::{
    CannyScratch, adaptive_threshold_gaussian, adaptive_threshold_mean, canny, canny_with_scratch,
    connected_components_4, threshold_binary, threshold_binary_inv, threshold_otsu,
    threshold_truncate,
};
pub use u8_canny::canny_u8;
pub use u8_color::{clahe_u8, histogram_u8, rgb_to_hsv_u8};
pub use u8_features::{
    bilateral_filter_u8, distance_transform_u8, fast9_detect_u8, warp_perspective_u8,
};
pub use u8_filters::{
    box_blur_3x3_u8, dilate_3x3_u8, erode_3x3_u8, gaussian_blur_3x3_u8, grayscale_u8,
    median_blur_3x3_u8, sobel_3x3_magnitude_u8,
};
pub use u8_harris::harris_corners_u8;
pub use u8_resize::{resize_bilinear_u8, resize_nearest_u8};
pub use u8ops::{ImageF32, ImageU8};
