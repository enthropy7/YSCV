//! Image processing primitives for yscv.
#![allow(unsafe_code)]

pub const CRATE_ID: &str = "yscv-imgproc";

#[path = "error.rs"]
mod error;
#[path = "ops/mod.rs"]
mod ops;
#[path = "shape.rs"]
mod shape;

pub use error::ImgProcError;

#[cfg(test)]
#[path = "proptest_tests.rs"]
mod proptest_tests;

// ── F32 tensor image operations ─────────────────────────────────────────
//
// Functions in this block operate on `Tensor` (f32) images, typically shaped
// `[H, W]` for grayscale or `[H, W, C]` for color. See individual function
// docs in their defining modules for full shape requirements.

/// Axis-aligned bounding box.
pub use ops::BBox;
/// 256-bit BRIEF binary descriptor.
pub use ops::BriefDescriptor;
/// Camera intrinsic parameters (fx, fy, cx, cy, distortion coefficients).
pub use ops::CameraIntrinsics;
/// Pre-allocated scratch buffers for [`canny_with_scratch`].
pub use ops::CannyScratch;
/// Statistics for a connected component region.
pub use ops::ComponentStats;
/// Contour represented as a sequence of (row, col) points.
pub use ops::Contour;
/// Configuration for Farneback dense optical flow.
pub use ops::FarnebackConfig;
/// Harris corner keypoint with location and response strength.
pub use ops::HarrisKeypoint;
/// Hough line in (rho, theta) form.
pub use ops::HoughLine;
/// Detected keypoint with location, response, and scale.
pub use ops::Keypoint;
/// Configuration for ORB feature detector.
pub use ops::OrbConfig;
/// ORB binary descriptor (256-bit).
pub use ops::OrbDescriptor;
/// Combined ORB keypoint + descriptor.
pub use ops::OrbFeature;
/// Properties of a labeled image region.
pub use ops::RegionProp;
/// Configuration for stereo block matching.
pub use ops::StereoConfig;
/// SURF descriptor.
pub use ops::SurfDescriptor;
/// SURF keypoint.
pub use ops::SurfKeypoint;
/// Method selection for template matching.
pub use ops::TemplateMatchMethod;
/// Result of template matching (location + score).
pub use ops::TemplateMatchResult;

/// Adaptive threshold using Gaussian-weighted local mean. Input: `[H, W]` grayscale.
pub use ops::adaptive_threshold_gaussian;
/// Adaptive threshold using box-filter local mean. Input: `[H, W]` grayscale.
pub use ops::adaptive_threshold_mean;
/// Gamma correction. Input: `[H, W]` or `[H, W, C]`.
pub use ops::adjust_gamma;
/// Logarithmic intensity adjustment.
pub use ops::adjust_log;
/// Approximate a contour with fewer vertices (Ramer-Douglas-Peucker).
pub use ops::approx_poly_dp;
/// Compute the arc length (perimeter) of a contour.
pub use ops::arc_length;
/// Edge-preserving bilateral filter. Input: `[H, W]` or `[H, W, C]`.
pub use ops::bilateral_filter;
/// Laplacian-of-Gaussian blob detection. Input: `[H, W]` grayscale.
pub use ops::blob_log;
/// Compute the bounding rectangle of a contour.
pub use ops::bounding_rect;
/// 3x3 box blur (mean filter). Input: `[H, W]` or `[H, W, C]` f32.
pub use ops::box_blur_3x3;
/// Build integral image (summed area table) for SURF.
pub use ops::build_integral_image;
/// Canny edge detector. Input: `[H, W]` grayscale f32, thresholds in image
/// intensity range. Returns binary edge map `[H, W]` with edges = 1.0.
pub use ops::canny;
/// Canny edge detection with pre-allocated scratch buffers for zero-alloc hot paths.
pub use ops::canny_with_scratch;
/// Contrast-limited adaptive histogram equalization. Input: `[H, W]` grayscale.
pub use ops::clahe;
/// Morphological closing (dilate then erode) with 3x3 structuring element.
/// Input: `[H, W]` grayscale f32.
pub use ops::closing_3x3;
/// Compute BRIEF binary descriptors at given keypoints. Input: `[H, W]` grayscale.
pub use ops::compute_brief;
/// Compute gradient orientation from Sobel gradients.
pub use ops::compute_gradient_orientation;
/// Compute SURF descriptors at detected keypoints.
pub use ops::compute_surf_descriptors;
/// 4-connected component labeling. Input: `[H, W]` binary.
pub use ops::connected_components_4;
/// Connected components with statistics (area, bounding box, centroid).
pub use ops::connected_components_with_stats;
/// Compute contour area (signed, using shoelace formula).
pub use ops::contour_area;
/// Compute the convex hull of a set of 2D points.
pub use ops::convex_hull;
/// Sub-pixel corner refinement.
pub use ops::corner_sub_pix;
/// Crop a region from an image tensor. Input: `[H, W, C]` or `[H, W]`.
pub use ops::crop;
/// Dense optical flow (Farneback method). Input: two consecutive `[H, W]` grayscale frames.
pub use ops::dense_optical_flow;
/// Detect ORB features (keypoints + descriptors). Input: `[H, W]` grayscale f32.
/// Returns `Vec<OrbFeature>`.
pub use ops::detect_orb;
/// Detect SURF keypoints using fast Hessian response.
pub use ops::detect_surf_keypoints;
/// Morphological dilation with arbitrary structuring element.
pub use ops::dilate;
/// Morphological dilation with 3x3 square structuring element.
/// Input: `[H, W]` grayscale f32.
pub use ops::dilate_3x3;
/// Euclidean distance transform. Input: `[H, W]` binary f32.
pub use ops::distance_transform;
/// Morphological erosion with arbitrary structuring element.
pub use ops::erode;
/// Morphological erosion with 3x3 square structuring element.
/// Input: `[H, W]` grayscale f32.
pub use ops::erode_3x3;
/// Farneback dense optical flow between two frames.
pub use ops::farneback_flow;
/// FAST corner detection (configurable arc length).
pub use ops::fast_corners;
/// FAST-9 corner detector. Input: `[H, W]` f32 grayscale.
/// Returns `Vec<Keypoint>` of detected corners above the threshold.
pub use ops::fast9_detect;
/// Generic 2D convolution with a custom kernel.
pub use ops::filter2d;
/// Find contours in a binary image. Input: `[H, W]` binary.
pub use ops::find_contours;
/// Fit an ellipse to a set of 2D points (least-squares).
pub use ops::fit_ellipse;
/// Flip image horizontally. Input: `[H, W, C]` or `[H, W]`.
pub use ops::flip_horizontal;
/// Flip image vertically. Input: `[H, W, C]` or `[H, W]`.
pub use ops::flip_vertical;
/// 3x3 Gaussian blur (sigma approx 0.85). Input: `[H, W]` or `[H, W, C]` f32.
pub use ops::gaussian_blur_3x3;
/// 5x5 Gaussian blur (sigma approx 1.0). Input: `[H, W]` or `[H, W, C]` f32.
pub use ops::gaussian_blur_5x5;
/// Gaussian image pyramid (successive half-resolution blurred images).
pub use ops::gaussian_pyramid;
/// Shi-Tomasi / Good Features to Track corner detector.
pub use ops::good_features_to_track;
/// Hamming distance between two binary descriptors.
pub use ops::hamming_distance;
/// Harris corner detector. Input: `[H, W]` grayscale f32.
/// Returns corner response map `[H, W]`.
pub use ops::harris_corners;
/// Compute 256-bin grayscale histogram. Input: `[H, W]` f32 with values in [0, 255].
pub use ops::histogram_256;
/// Histogram equalization. Input: `[H, W]` grayscale.
pub use ops::histogram_equalize;
/// HOG (Histogram of Oriented Gradients) cell descriptor.
pub use ops::hog_cell_descriptor;
/// Compute 4-point homography (direct linear transform).
pub use ops::homography_4pt;
/// Hough circle detection. Input: `[H, W]` edge/binary image.
pub use ops::hough_circles;
/// Hough line detection. Input: `[H, W]` edge/binary image.
pub use ops::hough_lines;
/// Convert HSV to RGB. Input/output: `[H, W, 3]` f32.
pub use ops::hsv_to_rgb;
/// Compute Hu's seven moment invariants from a binary/grayscale image.
pub use ops::hu_moments;
/// Inpaint missing regions using Telea's fast marching method.
pub use ops::inpaint_telea;
/// Compute integral image (summed area table). Input: `[H, W]`.
pub use ops::integral_image;
/// Convert CIE L*a*b* to RGB. Input/output: `[H, W, 3]`.
pub use ops::lab_to_rgb;
/// 3x3 Laplacian filter. Input: `[H, W]` grayscale f32.
pub use ops::laplacian_3x3;
/// Lucas-Kanade sparse optical flow.
pub use ops::lucas_kanade_optical_flow;
/// Match feature descriptors between two sets by brute-force nearest neighbor.
pub use ops::match_features;
/// Match two sets of SURF descriptors.
pub use ops::match_surf_descriptors;
/// 3x3 median blur. Input: `[H, W]` or `[H, W, C]` f32.
pub use ops::median_blur_3x3;
/// Median filter with arbitrary window size.
pub use ops::median_filter;
/// Minimum area rotated rectangle of a set of 2D points.
pub use ops::min_area_rect;
/// Morphological black-hat transform (closing - input).
pub use ops::morph_blackhat;
/// Morphological gradient (dilation - erosion) with 3x3 element.
pub use ops::morph_gradient_3x3;
/// Morphological top-hat transform (input - opening).
pub use ops::morph_tophat;
/// Non-maximum suppression on bounding boxes by IoU threshold.
pub use ops::nms;
/// Normalize image channels: `(pixel - mean) / std`. Input: `[H, W, C]` f32.
/// `mean` and `std` slices must have length equal to `C`.
pub use ops::normalize;
/// Morphological opening (erode then dilate) with 3x3 structuring element.
/// Input: `[H, W]` grayscale f32.
pub use ops::opening_3x3;
/// Compute ORB descriptors at given keypoints.
pub use ops::orb_descriptors;
/// Hamming distance between two ORB descriptors.
pub use ops::orb_hamming_distance;
/// Match ORB features between two sets.
pub use ops::orb_match;
/// Pad image borders with a constant value.
pub use ops::pad_constant;
/// Project 3D points to 2D using camera intrinsics.
pub use ops::project_points;
/// Estimate homography via RANSAC.
pub use ops::ransac_homography;
/// Compute region properties (area, centroid, bbox) from a labeled image.
pub use ops::region_props;
/// Remove small connected components below a size threshold.
pub use ops::remove_small_objects;
/// Resize using bilinear interpolation. Input: `[H, W, C]` or `[H, W]` f32.
pub use ops::resize_bilinear;
/// Resize using nearest-neighbor interpolation. Input: `[H, W, C]` or `[H, W]` f32.
pub use ops::resize_nearest;
/// Convert RGB to BGR channel order. Input/output: `[H, W, 3]`.
pub use ops::rgb_to_bgr;
/// Convert RGB to grayscale using ITU-R BT.601 luminance weights.
/// Input: `[H, W, 3]`, output: `[H, W]`.
pub use ops::rgb_to_grayscale;
/// Convert RGB to HSV color space. Input/output: `[H, W, 3]` f32.
pub use ops::rgb_to_hsv;
/// Convert RGB to CIE L*a*b*. Input/output: `[H, W, 3]`.
pub use ops::rgb_to_lab;
/// Convert RGB to YUV. Input/output: `[H, W, 3]`.
pub use ops::rgb_to_yuv;
/// Rotate image 90 degrees clockwise.
pub use ops::rotate90_cw;
/// 3x3 Scharr gradient (dx, dy). Input: `[H, W]` grayscale.
pub use ops::scharr_3x3_gradients;
/// 3x3 Scharr gradient magnitude. Input: `[H, W]` grayscale.
pub use ops::scharr_3x3_magnitude;
/// Compute SIFT-style descriptor at a keypoint.
pub use ops::sift_descriptor;
/// Match SIFT descriptors.
pub use ops::sift_match;
/// Morphological skeletonization. Input: `[H, W]` binary.
pub use ops::skeletonize;
/// 3x3 Sobel gradient (dx, dy). Input: `[H, W]` grayscale f32.
pub use ops::sobel_3x3_gradients;
/// 3x3 Sobel gradient magnitude. Input: `[H, W]` grayscale f32.
pub use ops::sobel_3x3_magnitude;
/// Stereo block matching for disparity estimation.
pub use ops::stereo_block_matching;
/// Template matching. Input: `[H, W]` image and `[TH, TW]` template.
pub use ops::template_match;
/// Binary threshold: output 1.0 where input >= thresh, else 0.0. Input: `[H, W]`.
pub use ops::threshold_binary;
/// Inverse binary threshold: output 1.0 where input < thresh. Input: `[H, W]`.
pub use ops::threshold_binary_inv;
/// Otsu's automatic threshold. Input: `[H, W]` grayscale with values in [0, 255].
pub use ops::threshold_otsu;
/// Truncate threshold: clamp values above thresh. Input: `[H, W]`.
pub use ops::threshold_truncate;
/// Triangulate 3D points from two camera views.
pub use ops::triangulate_points;
/// Undistort 2D points using camera intrinsics.
pub use ops::undistort_points;
/// Apply an affine warp transform. Input: `[H, W, C]` or `[H, W]`.
pub use ops::warp_affine;
/// Apply a perspective warp (homography). Input: `[H, W, C]` or `[H, W]`.
pub use ops::warp_perspective;
/// Watershed segmentation. Input: `[H, W]` grayscale, markers: `[H, W]` integer labels.
pub use ops::watershed;

// ── U8 image types and operations ───────────────────────────────────────

/// Detection result for drawing (bounding box, label, confidence).
pub use ops::DrawDetection;
/// HWC u8 image wrapper: `(width, height, channels, Vec<u8>)`.
pub use ops::ImageU8;
/// Alpha-blend two images with weights. Input: `[H, W, C]` f32.
pub use ops::add_weighted;
/// Bilateral filter on u8 image.
pub use ops::bilateral_filter_u8;
/// Bitwise AND of two binary/grayscale images.
pub use ops::bitwise_and;
/// Bitwise NOT of a binary/grayscale image.
pub use ops::bitwise_not;
/// Bitwise OR of two binary/grayscale images.
pub use ops::bitwise_or;
/// 3x3 box blur on u8 image.
pub use ops::box_blur_3x3_u8;
/// Canny edge detection on u8 grayscale image.
pub use ops::canny_u8;
/// Center crop to target size. Input: `[H, W, C]` or `[H, W]`.
pub use ops::center_crop;
/// Convert CHW tensor layout to HWC.
pub use ops::chw_to_hwc;
/// CLAHE on u8 grayscale image.
pub use ops::clahe_u8;
/// Random color jitter augmentation (brightness, contrast, saturation, hue).
pub use ops::color_jitter;
/// 3x3 dilation on u8 image.
pub use ops::dilate_3x3_u8;
/// Distance transform on u8 binary image.
pub use ops::distance_transform_u8;
/// Draw a circle on a u8 image.
pub use ops::draw_circle;
/// Draw detection boxes with labels on a u8 image.
pub use ops::draw_detections;
/// Draw a line on a u8 image.
pub use ops::draw_line;
/// Draw polylines on a u8 image.
pub use ops::draw_polylines;
/// Draw a rectangle on a u8 image.
pub use ops::draw_rect;
/// Draw text on a u8 image.
pub use ops::draw_text;
/// Draw scaled text on a u8 image.
pub use ops::draw_text_scaled;
/// Elastic deformation augmentation.
pub use ops::elastic_transform;
/// 3x3 erosion on u8 image.
pub use ops::erode_3x3_u8;
/// FAST-9 corner detection on u8 grayscale image.
pub use ops::fast9_detect_u8;
/// Fill a polygon region on a u8 image.
pub use ops::fill_poly;
/// 3x3 Gaussian blur on u8 image.
pub use ops::gaussian_blur_3x3_u8;
/// Add Gaussian noise to an image.
pub use ops::gaussian_noise;
/// Convert u8 RGB image to grayscale.
pub use ops::grayscale_u8;
/// Harris corner detection on u8 image.
pub use ops::harris_corners_u8;
/// Compute 256-bin histogram from u8 grayscale data.
pub use ops::histogram_u8;
/// Convert HWC tensor layout to CHW.
pub use ops::hwc_to_chw;
/// ImageNet-standard preprocessing (resize 256, center crop 224, normalize).
pub use ops::imagenet_preprocess;
/// Read an image file as `ImageU8`. Supports PNG, JPEG, BMP, GIF, TIFF, WebP.
pub use ops::imread;
/// Read an image file as grayscale `ImageU8`.
pub use ops::imread_gray;
/// Write an `ImageU8` to a file. Format inferred from extension (png, jpg, bmp, etc.).
pub use ops::imwrite;
/// Check if pixel values fall within a per-channel range.
pub use ops::in_range;
/// 3x3 median blur on u8 image.
pub use ops::median_blur_3x3_u8;
/// Normalize u8 image to [0, 1] f32 then apply mean/std normalization.
pub use ops::normalize_image;
/// Random crop augmentation.
pub use ops::random_crop;
/// Random erasing (cutout) augmentation.
pub use ops::random_erasing;
/// Random horizontal flip augmentation (50% probability).
pub use ops::random_horizontal_flip;
/// Random rotation augmentation.
pub use ops::random_rotation;
/// Random vertical flip augmentation (50% probability).
pub use ops::random_vertical_flip;
/// Rescale intensity to [0, 1] or a target range.
pub use ops::rescale_intensity;
/// Bilinear resize on u8 image.
pub use ops::resize_bilinear_u8;
/// Nearest-neighbor resize on u8 image.
pub use ops::resize_nearest_u8;
/// RGB to HSV conversion on u8 image.
pub use ops::rgb_to_hsv_u8;
/// 3x3 Sobel gradient magnitude on u8 image.
pub use ops::sobel_3x3_magnitude_u8;
/// Perspective warp on u8 image.
pub use ops::warp_perspective_u8;
/// Convert YUV to RGB. Input/output: `[H, W, 3]`.
pub use ops::yuv_to_rgb;

// ── F32 image wrapper and operations ────────────────────────────────────

/// HWC f32 image wrapper: `(width, height, channels, Vec<f32>)`.
pub use ops::ImageF32;
/// 3x3 box blur on f32 image.
pub use ops::box_blur_3x3_f32;
/// 3x3 dilation on f32 image.
pub use ops::dilate_3x3_f32;
/// 3x3 Gaussian blur on f32 image.
pub use ops::gaussian_blur_3x3_f32;
/// Convert f32 RGB image to grayscale.
pub use ops::grayscale_f32;
/// 3x3 Sobel gradient magnitude on f32 image.
pub use ops::sobel_3x3_f32;
/// Binary threshold on f32 image.
pub use ops::threshold_binary_f32;

#[path = "tests/mod.rs"]
#[cfg(test)]
mod tests;
