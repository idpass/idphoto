//! Face alignment via similarity transform from 5 detected landmarks
//! to the ArcFace reference template.
//!
//! When landmarks are available, this produces a properly aligned face image
//! where the eyes, nose, and mouth are at canonical positions, matching the
//! layout that ArcFace recognition models were trained on.

use image::{DynamicImage, RgbImage};

use crate::face_detector::FaceLandmarks;

/// Standard ArcFace 5-point reference template for a 112x112 output.
///
/// Points: right eye, left eye, nose tip, right mouth corner, left mouth corner.
/// These are the canonical positions from the InsightFace/ArcFace training pipeline.
const ARCFACE_TEMPLATE_112: [(f64, f64); 5] = [
    (38.2946, 51.6963), // right eye
    (73.5318, 51.5014), // left eye
    (56.0252, 71.7366), // nose tip
    (41.5493, 92.3655), // right mouth corner
    (70.7299, 92.2041), // left mouth corner
];

/// Alignment output size matching the ArcFace template.
const ALIGN_SIZE: u32 = 112;

/// 2x3 affine matrix [a, b, tx; c, d, ty] stored row-major as [a, b, tx, c, d, ty].
type AffineMatrix = [f64; 6];

/// Estimate a similarity transform (rotation + uniform scale + translation) mapping
/// `src` points to `dst` points using least squares.
///
/// A similarity transform has 4 degrees of freedom: (a, b, tx, ty) where
///   x' = a*x - b*y + tx
///   y' = b*x + a*y + ty
///
/// With 5 point pairs (10 equations, 4 unknowns), this is overdetermined.
/// We solve it via the normal equations (A^T A)^{-1} A^T b.
fn estimate_similarity(src: &[(f64, f64); 5], dst: &[(f64, f64); 5]) -> AffineMatrix {
    // Build the normal equation components for:
    //   [x_i, -y_i, 1, 0] [a ]   [x'_i]
    //   [y_i,  x_i, 0, 1] [b ] = [y'_i]
    //                      [tx]
    //                      [ty]
    let n = src.len();

    // A^T A is 4x4, A^T b is 4x1
    // But for a similarity transform, the structure is regular enough to solve directly.
    let mut sum_x2_y2 = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;

    let mut rhs_a = 0.0;
    let mut rhs_b = 0.0;
    let mut rhs_tx = 0.0;
    let mut rhs_ty = 0.0;

    for i in 0..n {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];

        sum_x2_y2 += sx * sx + sy * sy;
        sum_x += sx;
        sum_y += sy;

        rhs_a += sx * dx + sy * dy;
        rhs_b += sx * dy - sy * dx;
        rhs_tx += dx;
        rhs_ty += dy;
    }

    let nf = n as f64;

    // The normal equations for similarity transform:
    // [sum_x2_y2,  0,     sum_x, sum_y] [a ]   [rhs_a ]
    // [0,          sum_x2_y2, -sum_y, sum_x] [b ] = [rhs_b ]
    // [sum_x,     -sum_y, n,     0    ] [tx]   [rhs_tx]
    // [sum_y,      sum_x, 0,     n    ] [ty]   [rhs_ty]
    //
    // This can be reduced to two 2x2 systems by grouping (a,b) and (tx,ty).
    // Using the Schur complement:

    let det_main = sum_x2_y2 * nf - sum_x * sum_x - sum_y * sum_y;

    if det_main.abs() < 1e-10 {
        // Degenerate case: return identity
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    }

    // Solve via 4x4 system using Cramer's rule / direct inversion.
    // For numerical stability, use the block structure.

    // First solve for (a, b) from the reduced system:
    // After substituting tx, ty from rows 3,4 into rows 1,2:
    let a = (rhs_a * nf - sum_x * rhs_tx - sum_y * rhs_ty) / det_main;
    let b = (rhs_b * nf + sum_y * rhs_tx - sum_x * rhs_ty) / det_main;
    let tx = (rhs_tx - a * sum_x + b * sum_y) / nf;
    let ty = (rhs_ty - b * sum_x - a * sum_y) / nf;

    // Affine matrix: x' = a*x - b*y + tx
    //                y' = b*x + a*y + ty
    [a, -b, tx, b, a, ty]
}

/// Apply an affine transformation to an RGB image, producing an output of
/// `out_w x out_h` pixels using bilinear interpolation.
///
/// `matrix` is the forward transform (src -> dst). We invert it to sample
/// from the source for each output pixel.
fn warp_affine(image: &DynamicImage, matrix: &AffineMatrix, out_w: u32, out_h: u32) -> RgbImage {
    let rgb = image.to_rgb8();
    let (src_w, src_h) = (rgb.width() as f64, rgb.height() as f64);
    let mut output = RgbImage::new(out_w, out_h);

    // Invert the 2x3 affine matrix [a, -b, tx; b, a, ty]
    let (a, neg_b, tx) = (matrix[0], matrix[1], matrix[2]);
    let (b, a2, ty) = (matrix[3], matrix[4], matrix[5]);
    let _ = a2; // a2 == a for similarity transform

    let det = a * a + b * b; // a*a - (-b)*b = a^2 + b^2
    if det.abs() < 1e-10 {
        return output;
    }
    let inv_det = 1.0 / det;

    // Inverse: [a, b, -(a*tx + b*ty); -b, a, (b*tx - a*ty)] / det
    // But we use neg_b = -b in the matrix, so b_actual = -neg_b
    let ia = a * inv_det;
    let ib = -neg_b * inv_det; // neg_b is -b, so -neg_b = b, times inv_det
    let itx = -(ia * tx + ib * ty);
    let ic = neg_b * inv_det; // neg_b = -b, times inv_det = -b * inv_det
    let id = a * inv_det;
    let ity = -(ic * tx + id * ty);

    for out_y in 0..out_h {
        for out_x in 0..out_w {
            let dx = out_x as f64 + 0.5;
            let dy = out_y as f64 + 0.5;

            let sx = ia * dx + ib * dy + itx;
            let sy = ic * dx + id * dy + ity;

            // Bilinear interpolation
            let fx = sx - 0.5;
            let fy = sy - 0.5;

            let x0 = fx.floor() as i64;
            let y0 = fy.floor() as i64;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let wx = fx - x0 as f64;
            let wy = fy - y0 as f64;

            let sample = |px: i64, py: i64| -> [f64; 3] {
                if px < 0 || py < 0 || px >= src_w as i64 || py >= src_h as i64 {
                    return [0.0, 0.0, 0.0]; // black for out-of-bounds
                }
                let p = rgb.get_pixel(px as u32, py as u32);
                [p[0] as f64, p[1] as f64, p[2] as f64]
            };

            let p00 = sample(x0, y0);
            let p10 = sample(x1, y0);
            let p01 = sample(x0, y1);
            let p11 = sample(x1, y1);

            let mut pixel = [0u8; 3];
            for c in 0..3 {
                let v = p00[c] * (1.0 - wx) * (1.0 - wy)
                    + p10[c] * wx * (1.0 - wy)
                    + p01[c] * (1.0 - wx) * wy
                    + p11[c] * wx * wy;
                pixel[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            output.put_pixel(out_x, out_y, image::Rgb(pixel));
        }
    }

    output
}

/// Align a face using 5-point landmarks, producing a 112x112 RGB image
/// in the canonical ArcFace layout.
///
/// Returns `None` if the landmarks are degenerate (e.g., all at the same point).
pub(crate) fn align_face(image: &DynamicImage, landmarks: &FaceLandmarks) -> Option<DynamicImage> {
    let src = [
        landmarks.right_eye,
        landmarks.left_eye,
        landmarks.nose,
        landmarks.right_mouth,
        landmarks.left_mouth,
    ];

    // Sanity check: landmarks should not be degenerate
    let (min_x, max_x) = src.iter().fold((f64::MAX, f64::MIN), |(mn, mx), (x, _)| {
        (mn.min(*x), mx.max(*x))
    });
    let (min_y, max_y) = src.iter().fold((f64::MAX, f64::MIN), |(mn, mx), (_, y)| {
        (mn.min(*y), mx.max(*y))
    });
    if (max_x - min_x) < 1.0 || (max_y - min_y) < 1.0 {
        return None;
    }

    let matrix = estimate_similarity(&src, &ARCFACE_TEMPLATE_112);
    let aligned = warp_affine(image, &matrix, ALIGN_SIZE, ALIGN_SIZE);

    Some(DynamicImage::ImageRgb8(aligned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_preserves_points() {
        // Source and destination are the same -> should get near-identity matrix
        let points: [(f64, f64); 5] = [
            (38.0, 52.0),
            (74.0, 52.0),
            (56.0, 72.0),
            (42.0, 92.0),
            (71.0, 92.0),
        ];
        let m = estimate_similarity(&points, &points);
        // a should be ~1, b should be ~0
        assert!((m[0] - 1.0).abs() < 0.01, "a = {}", m[0]);
        assert!(m[1].abs() < 0.01, "-b = {}", m[1]);
        assert!(m[2].abs() < 0.5, "tx = {}", m[2]);
        assert!(m[3].abs() < 0.01, "b = {}", m[3]);
        assert!((m[4] - 1.0).abs() < 0.01, "a = {}", m[4]);
        assert!(m[5].abs() < 0.5, "ty = {}", m[5]);
    }

    #[test]
    fn scaled_transform() {
        // Source is 2x the template -> should get scale ~0.5
        let src: [(f64, f64); 5] = ARCFACE_TEMPLATE_112.map(|(x, y)| (x * 2.0, y * 2.0));
        let m = estimate_similarity(&src, &ARCFACE_TEMPLATE_112);
        assert!((m[0] - 0.5).abs() < 0.01, "a = {}", m[0]);
        assert!(m[1].abs() < 0.01, "-b = {}", m[1]);
    }

    #[test]
    fn align_face_produces_112x112() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(200, 200));
        let landmarks = FaceLandmarks {
            right_eye: (60.0, 80.0),
            left_eye: (140.0, 80.0),
            nose: (100.0, 120.0),
            right_mouth: (70.0, 150.0),
            left_mouth: (130.0, 150.0),
        };
        let aligned = align_face(&img, &landmarks).expect("alignment should succeed");
        assert_eq!(aligned.width(), 112);
        assert_eq!(aligned.height(), 112);
    }

    #[test]
    fn degenerate_landmarks_returns_none() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(200, 200));
        let landmarks = FaceLandmarks {
            right_eye: (100.0, 100.0),
            left_eye: (100.0, 100.0),
            nose: (100.0, 100.0),
            right_mouth: (100.0, 100.0),
            left_mouth: (100.0, 100.0),
        };
        assert!(align_face(&img, &landmarks).is_none());
    }
}
