use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// A contour as an ordered list of (x, y) boundary pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub points: Vec<(usize, usize)>,
}

/// Finds external contours in a binary single-channel `[H, W, 1]` image.
///
/// Pixels > 0.5 are foreground. Returns a list of contours where each contour
/// is an ordered sequence of border pixel coordinates using 8-connected
/// Moore boundary tracing.
pub fn find_contours(input: &Tensor) -> Result<Vec<Contour>, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let mut visited = vec![false; h * w];
    let mut contours = Vec::new();

    // 8-connected Moore neighborhood (clockwise from east)
    const DIRS: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] <= 0.5 || visited[y * w + x] {
                continue;
            }
            // Check if this is a border pixel (has at least one background 4-neighbor)
            let is_border = x == 0
                || x == w - 1
                || y == 0
                || y == h - 1
                || data[y * w + x - 1] <= 0.5
                || data[y * w + x + 1] <= 0.5
                || data[(y - 1) * w + x] <= 0.5
                || data[(y + 1) * w + x] <= 0.5;
            if !is_border {
                continue;
            }

            // Moore boundary trace
            let mut contour_points = Vec::new();
            let start = (x, y);
            let mut cur = start;
            let mut dir = 0usize; // start looking east
            let max_steps = h * w * 2;
            let mut steps = 0;

            loop {
                contour_points.push(cur);
                visited[cur.1 * w + cur.0] = true;

                let mut found = false;
                let start_dir = (dir + 5) % 8; // backtrack: turn right from where we came
                for i in 0..8 {
                    let d = (start_dir + i) % 8;
                    let (dx, dy) = DIRS[d];
                    let nx = cur.0 as i32 + dx;
                    let ny = cur.1 as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let (ux, uy) = (nx as usize, ny as usize);
                        if data[uy * w + ux] > 0.5 {
                            cur = (ux, uy);
                            dir = d;
                            found = true;
                            break;
                        }
                    }
                }

                if !found || cur == start || steps > max_steps {
                    break;
                }
                steps += 1;
            }

            if contour_points.len() >= 2 {
                contours.push(Contour {
                    points: contour_points,
                });
            }
        }
    }

    Ok(contours)
}

/// Computes the convex hull of 2D points using Andrew's monotone chain algorithm.
///
/// Input points are `(x, y)` pairs. Returns hull vertices in counter-clockwise order.
pub fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut pts: Vec<(f32, f32)> = akl_toussaint_filter(points);
    pts.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let n = pts.len();
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(2 * n);

    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Akl-Toussaint discard: drop points that provably cannot be hull vertices.
///
/// The four axis extremes span a convex quadrilateral, so every point strictly
/// inside it is a convex combination of hull members and can never be one
/// itself. Dropping them is exact — the returned set has the same convex hull —
/// and it turns the dominant `O(n log n)` sort into a linear pass over a much
/// smaller set. On dense blobs (a segmentation footprint, a filled contour)
/// this typically discards well over 90% of the input.
///
/// Points are kept when the strict-inside test fails, so anything on an edge of
/// the quadrilateral survives and float ties cannot discard a real vertex.
fn akl_toussaint_filter(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    // Small inputs: filtering costs more than it saves.
    if points.len() < 64 {
        return points.to_vec();
    }
    let mut left = points[0];
    let mut right = points[0];
    let mut bottom = points[0];
    let mut top = points[0];
    for &p in points {
        if p.0 < left.0 {
            left = p;
        }
        if p.0 > right.0 {
            right = p;
        }
        if p.1 < bottom.1 {
            bottom = p;
        }
        if p.1 > top.1 {
            top = p;
        }
    }
    let quad = [left, bottom, right, top];
    // Degenerate spans (a line, a point) leave the quadrilateral empty.
    let area = cross_2d(quad[0], quad[1], quad[2]) + cross_2d(quad[0], quad[2], quad[3]);
    if area.abs() < f32::EPSILON {
        return points.to_vec();
    }
    let orientation = area.signum();
    let strictly_inside = |p: (f32, f32)| -> bool {
        (0..4).all(|i| {
            let a = quad[i];
            let b = quad[(i + 1) % 4];
            cross_2d(a, b, p) * orientation > 0.0
        })
    };
    let mut kept: Vec<(f32, f32)> = Vec::with_capacity(points.len() / 8 + 16);
    kept.extend(points.iter().copied().filter(|&p| !strictly_inside(p)));
    kept
}

fn cross_2d(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Computes the minimum-area bounding rectangle for a set of 2D points
/// using the rotating calipers approach on the convex hull.
///
/// Returns `(center_x, center_y, width, height, angle_radians)`.
pub fn min_area_rect(points: &[(f32, f32)]) -> Option<(f32, f32, f32, f32, f32)> {
    let hull = convex_hull(points);
    if hull.len() < 2 {
        return hull.first().map(|p| (p.0, p.1, 0.0, 0.0, 0.0));
    }

    let n = hull.len();
    let mut best_area = f32::MAX;
    let mut best_rect = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

    for i in 0..n {
        let j = (i + 1) % n;
        let edge_x = hull[j].0 - hull[i].0;
        let edge_y = hull[j].1 - hull[i].1;
        let edge_len = (edge_x * edge_x + edge_y * edge_y).sqrt();
        if edge_len < 1e-12 {
            continue;
        }
        let ux = edge_x / edge_len;
        let uy = edge_y / edge_len;

        let mut min_proj = f32::MAX;
        let mut max_proj = f32::MIN;
        let mut min_perp = f32::MAX;
        let mut max_perp = f32::MIN;

        for &p in &hull {
            let dx = p.0 - hull[i].0;
            let dy = p.1 - hull[i].1;
            let proj = dx * ux + dy * uy;
            let perp = -dx * uy + dy * ux;
            min_proj = min_proj.min(proj);
            max_proj = max_proj.max(proj);
            min_perp = min_perp.min(perp);
            max_perp = max_perp.max(perp);
        }

        let width = max_proj - min_proj;
        let height = max_perp - min_perp;
        let area = width * height;
        if area < best_area {
            best_area = area;
            let mid_proj = (min_proj + max_proj) * 0.5;
            let mid_perp = (min_perp + max_perp) * 0.5;
            let cx = hull[i].0 + ux * mid_proj - uy * mid_perp;
            let cy = hull[i].1 + uy * mid_proj + ux * mid_perp;
            let angle = uy.atan2(ux);
            best_rect = (cx, cy, width, height, angle);
        }
    }

    Some(best_rect)
}

/// Computes a 3x3 homography matrix from 4 source/destination point correspondences
/// using the Direct Linear Transform (DLT) algorithm.
///
/// `src` and `dst` must each contain exactly 4 points `(x, y)`.
/// Returns the 9-element matrix in row-major order.
pub fn homography_4pt(
    src: &[(f32, f32); 4],
    dst: &[(f32, f32); 4],
) -> Result<[f32; 9], ImgProcError> {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];

    for i in 0..4 {
        let (sx, sy) = (src[i].0 as f64, src[i].1 as f64);
        let (dx, dy) = (dst[i].0 as f64, dst[i].1 as f64);
        let r = i * 2;
        a[r] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy];
        b[r] = dx;
        a[r + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy];
        b[r + 1] = dy;
    }

    let h =
        solve_8x8(&a, &b).ok_or(ImgProcError::InvalidOutputDimensions { out_h: 0, out_w: 0 })?;

    Ok([
        h[0] as f32,
        h[1] as f32,
        h[2] as f32,
        h[3] as f32,
        h[4] as f32,
        h[5] as f32,
        h[6] as f32,
        h[7] as f32,
        1.0,
    ])
}

#[allow(clippy::needless_range_loop)]
fn solve_8x8(a: &[[f64; 8]; 8], b: &[f64; 8]) -> Option<[f64; 8]> {
    let mut m = [[0.0f64; 9]; 8];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = a[i][j];
        }
        m[i][8] = b[i];
    }

    for col in 0..8 {
        let mut pivot = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                pivot = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(pivot, col);
        }
        let diag = m[col][col];
        for j in col..9 {
            m[col][j] /= diag;
        }
        for row in 0..8 {
            if row != col {
                let factor = m[row][col];
                for j in col..9 {
                    m[row][j] -= factor * m[col][j];
                }
            }
        }
    }

    let mut result = [0.0f64; 8];
    for i in 0..8 {
        result[i] = m[i][8];
    }
    Some(result)
}

/// RANSAC-based homography estimation from point correspondences.
///
/// Iteratively samples 4-point subsets, computes candidate homographies via DLT,
/// and selects the model with the most inliers under `inlier_threshold`.
/// Returns `(homography [9], inlier_mask)`.
pub fn ransac_homography(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    iterations: usize,
    inlier_threshold: f32,
    rng_seed: u64,
) -> Option<([f32; 9], Vec<bool>)> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let n = src.len();
    let mut best_h = [0.0f32; 9];
    let mut best_inliers: Vec<bool> = vec![false; n];
    let mut best_count = 0usize;
    let mut rng_state = rng_seed;

    for _ in 0..iterations {
        let mut indices = [0usize; 4];
        for slot in &mut indices {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (rng_state >> 33) as usize % n;
        }
        let src4: [(f32, f32); 4] = [
            src[indices[0]],
            src[indices[1]],
            src[indices[2]],
            src[indices[3]],
        ];
        let dst4: [(f32, f32); 4] = [
            dst[indices[0]],
            dst[indices[1]],
            dst[indices[2]],
            dst[indices[3]],
        ];
        let h = match homography_4pt(&src4, &dst4) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let mut inliers = vec![false; n];
        let mut count = 0;
        for i in 0..n {
            let (sx, sy) = src[i];
            let denom = h[6] * sx + h[7] * sy + h[8];
            if denom.abs() < 1e-12 {
                continue;
            }
            let px = (h[0] * sx + h[1] * sy + h[2]) / denom;
            let py = (h[3] * sx + h[4] * sy + h[5]) / denom;
            let err = ((px - dst[i].0).powi(2) + (py - dst[i].1).powi(2)).sqrt();
            if err < inlier_threshold {
                inliers[i] = true;
                count += 1;
            }
        }
        if count > best_count {
            best_count = count;
            best_h = h;
            best_inliers = inliers;
        }
    }

    if best_count >= 4 {
        Some((best_h, best_inliers))
    } else {
        None
    }
}

/// Fits an axis-aligned ellipse to a set of 2D points using the method of moments.
///
/// Returns `(center_x, center_y, semi_axis_a, semi_axis_b, rotation_angle_radians)`.
pub fn fit_ellipse(points: &[(f32, f32)]) -> Option<(f32, f32, f32, f32, f32)> {
    if points.len() < 5 {
        return None;
    }
    let n = points.len() as f32;
    let cx: f32 = points.iter().map(|p| p.0).sum::<f32>() / n;
    let cy: f32 = points.iter().map(|p| p.1).sum::<f32>() / n;

    let mut cov_xx = 0.0f32;
    let mut cov_xy = 0.0f32;
    let mut cov_yy = 0.0f32;
    for &(x, y) in points {
        let dx = x - cx;
        let dy = y - cy;
        cov_xx += dx * dx;
        cov_xy += dx * dy;
        cov_yy += dy * dy;
    }
    cov_xx /= n;
    cov_xy /= n;
    cov_yy /= n;

    let trace = cov_xx + cov_yy;
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    let disc = (trace * trace / 4.0 - det).max(0.0).sqrt();
    let lambda1 = trace / 2.0 + disc;
    let lambda2 = (trace / 2.0 - disc).max(1e-12);

    let angle = cov_xy.atan2(lambda1 - cov_yy);
    let a = (2.0 * lambda1).sqrt();
    let b = (2.0 * lambda2).sqrt();

    Some((cx, cy, a, b, angle))
}

/// Least-squares circle fit (Kåsa method).
///
/// Solves the linear system arising from `x² + y² = c₀x + c₁y + c₂` in
/// closed form (3×3 normal equations, f64 accumulation for stability) and
/// returns `(center_x, center_y, radius)`. Returns `None` for fewer than
/// three points or a degenerate (collinear) configuration.
pub fn fit_circle(points: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    // Normal equations AᵀA s = Aᵀb with A = [x, y, 1], b = x² + y².
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for &(px, py) in points {
        let (x, y) = (f64::from(px), f64::from(py));
        let row = [x, y, 1.0];
        let b = x * x + y * y;
        for (i, &ri) in row.iter().enumerate() {
            atb[i] += ri * b;
            for (v, &rj) in ata[i].iter_mut().zip(&row) {
                *v += ri * rj;
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut m = [[0.0f64; 4]; 3];
    for (mr, (ar, &bv)) in m.iter_mut().zip(ata.iter().zip(&atb)) {
        mr[..3].copy_from_slice(ar);
        mr[3] = bv;
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&r1, &r2| m[r1][col].abs().total_cmp(&m[r2][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let piv_row = m[col];
        for (row, mr) in m.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let f = mr[col] / piv_row[col];
            for (v, &pv) in mr.iter_mut().zip(&piv_row).skip(col) {
                *v -= f * pv;
            }
        }
    }
    let c0 = m[0][3] / m[0][0];
    let c1 = m[1][3] / m[1][1];
    let c2 = m[2][3] / m[2][2];
    let cx = c0 / 2.0;
    let cy = c1 / 2.0;
    let r_sq = c2 + cx * cx + cy * cy;
    if r_sq <= 0.0 || !r_sq.is_finite() {
        return None;
    }
    Some((cx as f32, cy as f32, r_sq.sqrt() as f32))
}

/// Douglas-Peucker contour approximation.
///
/// Simplifies a polyline by recursively removing points within `epsilon` distance
/// from the line segment connecting endpoints.
pub fn approx_poly_dp(contour: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if contour.len() <= 2 {
        return contour.to_vec();
    }
    let n = contour.len();
    let (first, last) = (contour[0], contour[n - 1]);

    let mut max_dist = 0.0f32;
    let mut max_idx = 0;
    for (i, &pt) in contour.iter().enumerate().skip(1).take(n - 2) {
        let d = point_line_dist_f32(pt, first, last);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        let mut left = approx_poly_dp(&contour[..=max_idx], epsilon);
        let right = approx_poly_dp(&contour[max_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

/// Statistics for a single connected component.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentStats {
    pub label: usize,
    pub area: usize,
    pub bbox: (usize, usize, usize, usize), // (x, y, w, h)
    pub centroid: (f32, f32),               // (cx, cy)
}

/// Properties for a labelled region.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionProp {
    pub label: usize,
    pub area: usize,
    pub centroid: (f32, f32),
    pub bbox: (usize, usize, usize, usize), // (x, y, w, h)
    pub perimeter: f32,
}

/// Maximal horizontal span of foreground pixels: columns `[start, end)` of `row`.
struct Run {
    row: usize,
    start: usize,
    end: usize,
    label: u32,
}

/// Disjoint set over provisional run labels: union by size, path halving.
struct RunUnion {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl RunUnion {
    fn with_capacity(runs: usize) -> Self {
        Self {
            parent: Vec::with_capacity(runs),
            size: Vec::with_capacity(runs),
        }
    }

    fn make(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.size.push(1);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

/// Shared implementation of the two labelling entry points.
///
/// Works on horizontal runs, not pixels: a row is scanned once into maximal
/// foreground spans, each span joins the spans it touches in the row above, and
/// a disjoint set resolves the chains. Cost then follows the number of spans,
/// and a solid object is a couple of spans per row however wide it is.
///
/// Statistics come out of the same spans in closed form — a span of length `n`
/// starting at `s` adds `n` to the area and `(2s + n - 1)·n/2` to the x-moment,
/// both exact in `f64` at any image size that fits in memory — so individual
/// pixels are never walked at all.
///
/// Labels follow the raster order of each component's first pixel, which is
/// what a flood fill produces and what callers rely on.
fn label_components(
    img: &Tensor,
    diagonal: bool,
) -> Result<(Tensor, Vec<ComponentStats>), ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = img.data();
    // Diagonal contact widens the overlap test by one column on each side.
    let slack = usize::from(diagonal);

    let mut runs: Vec<Run> = Vec::new();
    let mut union = RunUnion::with_capacity(w.max(1));
    let mut previous_row = 0..0usize;

    for row in 0..h {
        let line = &data[row * w..(row + 1) * w];
        let row_start = runs.len();
        let mut above = previous_row.start;

        let mut col = 0usize;
        while col < w {
            if line[col] <= 0.5 {
                col += 1;
                continue;
            }
            let start = col;
            while col < w && line[col] > 0.5 {
                col += 1;
            }
            let end = col;

            let label = union.make();
            // Runs of both rows are sorted by start, so the window of spans
            // above that can touch this one only ever moves forward.
            while above < previous_row.end && runs[above].end + slack <= start {
                above += 1;
            }
            let mut touching = above;
            while touching < previous_row.end && runs[touching].start < end + slack {
                let other = runs[touching].label;
                union.union(label, other);
                touching += 1;
            }
            runs.push(Run {
                row,
                start,
                end,
                label,
            });
        }
        previous_row = row_start..runs.len();
    }

    // Final labels follow the first run of each component, which is its first
    // pixel in raster order.
    let mut final_label = vec![0u32; runs.len()];
    let mut stats_list: Vec<ComponentStats> = Vec::new();
    let mut moments: Vec<(f64, f64)> = Vec::new();
    let mut label_data = vec![0.0f32; h * w];

    for index in 0..runs.len() {
        let root = union.find(runs[index].label) as usize;
        let Run {
            row, start, end, ..
        } = runs[index];
        let length = end - start;

        if final_label[root] == 0 {
            stats_list.push(ComponentStats {
                label: stats_list.len() + 1,
                area: 0,
                bbox: (start, row, 0, 0),
                centroid: (0.0, 0.0),
            });
            moments.push((0.0, 0.0));
            final_label[root] = stats_list.len() as u32;
        }
        let label = final_label[root];
        let stats = &mut stats_list[label as usize - 1];
        let (sum_x, sum_y) = &mut moments[label as usize - 1];

        stats.area += length;
        // Held as (min_x, min_y, max_x, max_y) until the end. Rows are visited
        // in order, so min_y is already final from the component's first run.
        let (min_x, min_y, max_x, max_y) = stats.bbox;
        stats.bbox = (min_x.min(start), min_y, max_x.max(end - 1), max_y.max(row));
        *sum_x += (start + end - 1) as f64 * length as f64 / 2.0;
        *sum_y += (row * length) as f64;

        label_data[row * w + start..row * w + end].fill(label as f32);
    }

    for (stats, (sum_x, sum_y)) in stats_list.iter_mut().zip(&moments) {
        let (min_x, min_y, max_x, max_y) = stats.bbox;
        stats.bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        let area = stats.area as f64;
        stats.centroid = ((sum_x / area) as f32, (sum_y / area) as f32);
    }

    Ok((Tensor::from_vec(vec![h, w, 1], label_data)?, stats_list))
}

/// Connected-component labelling with per-component statistics.
///
/// Input: single-channel binary image `[H, W, 1]` (pixels > 0.5 are foreground).
/// Returns `(label_image, stats)` where `label_image` has shape `[H, W, 1]` with
/// label 0 for background and labels 1..N for components. 4-connectivity.
pub fn connected_components_with_stats(
    img: &Tensor,
) -> Result<(Tensor, Vec<ComponentStats>), ImgProcError> {
    label_components(img, false)
}

/// Connected-component labelling with per-component statistics, 8-connectivity.
///
/// Same contract as [`connected_components_with_stats`], but diagonal
/// neighbours join a component (OpenCV `connectivity=8` semantics) — the
/// usual choice for object masks, where a 1-px diagonal bridge still means
/// one object.
pub fn connected_components_with_stats_8(
    img: &Tensor,
) -> Result<(Tensor, Vec<ComponentStats>), ImgProcError> {
    label_components(img, true)
}

/// Minimum enclosing circle of a point set (Welzl's algorithm, exact).
///
/// Returns `((cx, cy), radius)`, or `None` for an empty input. Runs in
/// expected O(n) after a deterministic move-to-front shuffle; internal math
/// is f64 for stability on clustered points.
pub fn min_enclosing_circle(points: &[(f32, f32)]) -> Option<((f32, f32), f32)> {
    if points.is_empty() {
        return None;
    }
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x, y)| (f64::from(x), f64::from(y)))
        .collect();
    // Deterministic pseudo-shuffle (splitmix-стиль) — рандомизация Welzl
    // без внешнего RNG, воспроизводимо между запусками.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..pts.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        pts.swap(i, j);
    }

    fn circle_from2(a: (f64, f64), b: (f64, f64)) -> ((f64, f64), f64) {
        let c = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        (c, dist(c, a))
    }
    fn circle_from3(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Option<((f64, f64), f64)> {
        let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
        if d.abs() < 1e-12 {
            return None; // коллинеарны
        }
        let a2 = a.0 * a.0 + a.1 * a.1;
        let b2 = b.0 * b.0 + b.1 * b.1;
        let c2 = c.0 * c.0 + c.1 * c.1;
        let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
        let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
        let center = (ux, uy);
        Some((center, dist(center, a)))
    }
    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
        ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)).sqrt()
    }
    fn inside(c: ((f64, f64), f64), p: (f64, f64)) -> bool {
        dist(c.0, p) <= c.1 * (1.0 + 1e-10) + 1e-12
    }

    // Итеративный Welzl (move-to-front): без рекурсии, поддержка до 3 опор.
    let mut circle: ((f64, f64), f64) = (pts[0], 0.0);
    for i in 1..pts.len() {
        if inside(circle, pts[i]) {
            continue;
        }
        circle = (pts[i], 0.0);
        for j in 0..i {
            if inside(circle, pts[j]) {
                continue;
            }
            circle = circle_from2(pts[i], pts[j]);
            for k in 0..j {
                if inside(circle, pts[k]) {
                    continue;
                }
                if let Some(c3) = circle_from3(pts[i], pts[j], pts[k]) {
                    circle = c3;
                }
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    Some(((circle.0.0 as f32, circle.0.1 as f32), circle.1 as f32))
}

/// Compute region properties from a label image.
///
/// Input: label image `[H, W, 1]` (e.g. from `connected_components_with_stats`).
/// For each non-zero label, computes area, centroid, bounding box, and perimeter.
/// Perimeter counts pixels that are adjacent to a different label or to the image edge.
pub fn region_props(labels: &Tensor) -> Result<Vec<RegionProp>, ImgProcError> {
    let (h, w, c) = hwc_shape(labels)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = labels.data();

    // Find all unique non-zero labels
    let mut max_label = 0u32;
    for &v in data.iter() {
        let l = v as u32;
        if l > max_label {
            max_label = l;
        }
    }
    if max_label == 0 {
        return Ok(Vec::new());
    }

    // Accumulators per label (1-indexed, slot 0 unused)
    let n = max_label as usize;
    let mut area = vec![0usize; n + 1];
    let mut sum_x = vec![0.0f64; n + 1];
    let mut sum_y = vec![0.0f64; n + 1];
    let mut min_x = vec![usize::MAX; n + 1];
    let mut max_x = vec![0usize; n + 1];
    let mut min_y = vec![usize::MAX; n + 1];
    let mut max_y = vec![0usize; n + 1];
    let mut perim = vec![0usize; n + 1];

    for y in 0..h {
        for x in 0..w {
            let l = data[y * w + x] as u32;
            if l == 0 {
                continue;
            }
            let li = l as usize;
            area[li] += 1;
            sum_x[li] += x as f64;
            sum_y[li] += y as f64;
            if x < min_x[li] {
                min_x[li] = x;
            }
            if x > max_x[li] {
                max_x[li] = x;
            }
            if y < min_y[li] {
                min_y[li] = y;
            }
            if y > max_y[li] {
                max_y[li] = y;
            }

            // Boundary pixel: adjacent to a different label or at the image edge
            let is_boundary = x == 0
                || x == w - 1
                || y == 0
                || y == h - 1
                || data[y * w + (x - 1)] as u32 != l
                || data[y * w + (x + 1)] as u32 != l
                || data[(y - 1) * w + x] as u32 != l
                || data[(y + 1) * w + x] as u32 != l;
            if is_boundary {
                perim[li] += 1;
            }
        }
    }

    let mut props = Vec::new();
    for li in 1..=n {
        if area[li] == 0 {
            continue;
        }
        props.push(RegionProp {
            label: li,
            area: area[li],
            centroid: (
                (sum_x[li] / area[li] as f64) as f32,
                (sum_y[li] / area[li] as f64) as f32,
            ),
            bbox: (
                min_x[li],
                min_y[li],
                max_x[li] - min_x[li] + 1,
                max_y[li] - min_y[li] + 1,
            ),
            perimeter: perim[li] as f32,
        });
    }

    Ok(props)
}

/// Computes the 7 Hu invariant moments from a single-channel `[H, W, 1]` image.
///
/// Hu moments are invariant to translation, scale, and rotation. They are
/// derived from the normalised central moments of the image.
pub fn hu_moments(img: &Tensor) -> Result<[f64; 7], ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = img.data();

    // Raw moments m00, m10, m01
    let mut m00 = 0.0f64;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let v = data[y * w + x] as f64;
            m00 += v;
            m10 += x as f64 * v;
            m01 += y as f64 * v;
        }
    }
    if m00.abs() < 1e-15 {
        return Ok([0.0; 7]);
    }
    let cx = m10 / m00;
    let cy = m01 / m00;

    // Central moments up to order 3
    let mut mu20 = 0.0f64;
    let mut mu02 = 0.0f64;
    let mut mu11 = 0.0f64;
    let mut mu30 = 0.0f64;
    let mut mu03 = 0.0f64;
    let mut mu21 = 0.0f64;
    let mut mu12 = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let v = data[y * w + x] as f64;
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            mu20 += dx * dx * v;
            mu02 += dy * dy * v;
            mu11 += dx * dy * v;
            mu30 += dx * dx * dx * v;
            mu03 += dy * dy * dy * v;
            mu21 += dx * dx * dy * v;
            mu12 += dx * dy * dy * v;
        }
    }

    // Normalised central moments: eta_pq = mu_pq / m00^((p+q)/2 + 1)
    let n20 = mu20 / m00.powf(2.0);
    let n02 = mu02 / m00.powf(2.0);
    let n11 = mu11 / m00.powf(2.0);
    let n30 = mu30 / m00.powf(2.5);
    let n03 = mu03 / m00.powf(2.5);
    let n21 = mu21 / m00.powf(2.5);
    let n12 = mu12 / m00.powf(2.5);

    // Hu's 7 invariants
    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11 * n11;
    let h3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let h4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    let h5 = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));
    let h6 = (n20 - n02) * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
        + 4.0 * n11 * (n30 + n12) * (n21 + n03);
    let h7 = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));

    Ok([h1, h2, h3, h4, h5, h6, h7])
}

fn point_line_dist_f32(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }
    let cross = ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs();
    cross / len_sq.sqrt()
}

/// Computes the area of a polygon defined by its vertices using the Shoelace formula.
///
/// The contour should be an ordered sequence of `(x, y)` vertex coordinates.
/// Returns the absolute area of the polygon.
pub fn contour_area(contour: &[(usize, usize)]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }
    let n = contour.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = (contour[i].0 as f64, contour[i].1 as f64);
        let (x2, y2) = (contour[j].0 as f64, contour[j].1 as f64);
        sum += x1 * y2 - x2 * y1;
    }
    sum.abs() / 2.0
}

/// Computes the perimeter (arc length) of a contour.
///
/// Sums the Euclidean distances between consecutive points.
/// If `closed` is true, also adds the distance from the last point back to the first.
pub fn arc_length(contour: &[(usize, usize)], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0f64;
    for i in 0..contour.len() - 1 {
        let dx = contour[i + 1].0 as f64 - contour[i].0 as f64;
        let dy = contour[i + 1].1 as f64 - contour[i].1 as f64;
        length += (dx * dx + dy * dy).sqrt();
    }
    if closed {
        let dx = contour[0].0 as f64 - contour[contour.len() - 1].0 as f64;
        let dy = contour[0].1 as f64 - contour[contour.len() - 1].1 as f64;
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// Computes the axis-aligned bounding rectangle of a contour.
///
/// Returns `(x, y, width, height)` where `(x, y)` is the top-left corner.
pub fn bounding_rect(contour: &[(usize, usize)]) -> (usize, usize, usize, usize) {
    if contour.is_empty() {
        return (0, 0, 0, 0);
    }
    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for &(x, y) in contour {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x - min_x, max_y - min_y)
}
