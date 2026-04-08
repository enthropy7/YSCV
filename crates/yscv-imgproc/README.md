# yscv-imgproc

159 SIMD-accelerated image processing functions (free `pub fn` items in `src/ops/`) for f32 and u8 images. Covers everything from basic filters to feature detection.

```rust
use yscv_imgproc::*;

let img = imread("photo.jpg")?;
let edges = canny(&rgb_to_grayscale(&img)?, 50.0, 150.0)?;
let blurred = gaussian_blur_3x3(&img)?;
let resized = resize_bilinear(&img, 224, 224)?;
```

## Operations

| Category | Functions |
|----------|-----------|
| **Color** | rgb_to_grayscale, rgb_to_hsv, rgb_to_lab, rgb_to_yuv, bgr_to_rgb |
| **Filter** | box_blur, gaussian_blur, median_blur, bilateral_filter, sobel |
| **Morphology** | dilate, erode, opening, closing, gradient, tophat, blackhat, skeletonize |
| **Edge** | canny, sobel_3x3_gradients, sobel_3x3_magnitude |
| **Features** | fast9, harris_corners, sift, surf, orb, brief, template_match |
| **Geometry** | resize (nearest/bilinear), flip, rotate90, warp_affine, warp_perspective |
| **Threshold** | binary, binary_inv, truncate, adaptive, otsu |
| **Contour** | find_contours, contour_area, bounding_rect, convex_hull, approx_poly |
| **Drawing** | draw_rect, draw_circle, draw_line, draw_text, fill_poly |
| **Histogram** | equalize_hist, clahe, calc_hist |
| **Stereo** | disparity_map, optical_flow (Farneback) |
| **I/O** | imread, imwrite (PNG, JPEG, BMP) |

## Performance

All hot paths have NEON (aarch64) and SSE2/AVX (x86_64) SIMD. GCD parallel dispatch on macOS, rayon on Linux/Windows.

## Tests

225 tests. Criterion benchmarks for grayscale, resize, blur, morphology, sobel.
