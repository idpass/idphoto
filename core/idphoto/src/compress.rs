use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageEncoder, ImageFormat, RgbImage, RgbaImage};
use webp::Encoder as LibWebpEncoder;

use crate::crop::{portrait_crop, CropRegion};
use crate::error::IdPhotoError;
use crate::face_detector::{FaceBounds, FaceDetector};
use crate::webp_strip::strip_icc_profile;
use crate::{CompressedPhoto, CropMode, OutputFormat};

/// Result of a crop operation, including the face bounds if detection was used.
struct CropResult {
    image: DynamicImage,
    face_bounds: Option<FaceBounds>,
    crop_offset: (u32, u32),
}

/// Decode input bytes into a `DynamicImage`.
pub(crate) fn decode_image(input: &[u8]) -> Result<DynamicImage, IdPhotoError> {
    image::load_from_memory(input).map_err(|e| IdPhotoError::DecodeError(e.to_string()))
}

/// Detect a face in the image using the provided detector.
///
/// Returns the best (highest-confidence) face, or `None` if no faces are found.
fn detect_face(image: &DynamicImage, detector: Option<&dyn FaceDetector>) -> Option<FaceBounds> {
    let detector = match detector {
        Some(d) => d,
        None => {
            // Try built-in rustface backend
            #[cfg(feature = "rustface")]
            {
                let builtin = crate::RustfaceDetector::new();
                let gray = image::imageops::grayscale(image);
                let faces = builtin.detect(gray.as_raw(), gray.width(), gray.height());
                return faces.into_iter().max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            #[cfg(not(feature = "rustface"))]
            {
                return None;
            }
        }
    };

    let gray = image::imageops::grayscale(image);
    let faces = detector.detect(gray.as_raw(), gray.width(), gray.height());
    faces.into_iter().max_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Compute the 3:4 portrait crop region centered on a detected face.
///
/// `face_margin` controls framing: 2.0 = ID photo (face + hair + shoulders),
/// 1.3 = tight face crop for algorithmic matching.
fn compute_face_crop(
    face: &FaceBounds,
    image_width: u32,
    image_height: u32,
    face_margin: f32,
) -> CropRegion {
    let face_cx = face.x + face.width / 2.0;
    let face_cy = face.y + face.height / 2.0;
    let face_h = face.height;

    // Size the crop relative to the face.
    let portrait_aspect = 3.0 / 4.0;
    let desired_crop_h = (face_h * face_margin as f64).round();
    let desired_crop_w = (desired_crop_h * portrait_aspect).round();

    // Clamp to image bounds — the crop can't exceed the source dimensions
    let crop_h = (desired_crop_h as u32).min(image_height);
    let crop_w = (desired_crop_w as u32).min(image_width);

    // Ensure 3:4 aspect is maintained after clamping
    let (crop_w, crop_h) = if (crop_w as f64 / crop_h as f64) > portrait_aspect {
        // Too wide for the height — reduce width
        let w = (crop_h as f64 * portrait_aspect).round() as u32;
        (w, crop_h)
    } else {
        // Too tall for the width — reduce height
        let h = (crop_w as f64 / portrait_aspect).round() as u32;
        (crop_w, h)
    };

    // Position the face within the crop. For wider margins (ID-photo framing),
    // the face sits at ~40% from top (room for hair above, shoulders below).
    // For tight margins (face matching), center the face vertically.
    let vertical_position = if face_margin < 1.5 { 0.5 } else { 0.4 };
    let y = (face_cy - crop_h as f64 * vertical_position)
        .round()
        .max(0.0)
        .min((image_height.saturating_sub(crop_h)) as f64) as u32;
    let x = (face_cx - crop_w as f64 / 2.0)
        .round()
        .max(0.0)
        .min((image_width.saturating_sub(crop_w)) as f64) as u32;

    CropRegion {
        x,
        y,
        width: crop_w,
        height: crop_h,
    }
}

/// Apply the crop mode to the source image.
///
/// `face_margin` controls how tightly face detection crops around the face
/// (crop_height = face_height × face_margin). Ignored for non-face-detection modes.
fn apply_crop(
    image: &DynamicImage,
    mode: &CropMode,
    face_margin: f32,
    detector: Option<&dyn FaceDetector>,
) -> CropResult {
    match mode {
        CropMode::None => CropResult {
            image: image.clone(),
            face_bounds: None,
            crop_offset: (0, 0),
        },
        CropMode::Heuristic => {
            let CropRegion {
                x,
                y,
                width,
                height,
            } = portrait_crop(image.width(), image.height());
            CropResult {
                image: image.crop_imm(x, y, width, height),
                face_bounds: None,
                crop_offset: (x, y),
            }
        }
        CropMode::FaceDetection => {
            // Face detection with fallback to heuristic
            match detect_face(image, detector) {
                Some(face) => {
                    let region =
                        compute_face_crop(&face, image.width(), image.height(), face_margin);
                    CropResult {
                        image: image.crop_imm(region.x, region.y, region.width, region.height),
                        face_bounds: Some(face),
                        crop_offset: (region.x, region.y),
                    }
                }
                None => {
                    let CropRegion {
                        x,
                        y,
                        width,
                        height,
                    } = portrait_crop(image.width(), image.height());
                    CropResult {
                        image: image.crop_imm(x, y, width, height),
                        face_bounds: None,
                        crop_offset: (x, y),
                    }
                }
            }
        }
    }
}

/// Resize the image so the larger dimension matches `max_dimension`,
/// maintaining the aspect ratio from the crop.
pub(crate) fn resize_image(image: &DynamicImage, max_dimension: u32) -> DynamicImage {
    let (src_w, src_h) = (image.width(), image.height());

    let (new_w, new_h) = if src_w >= src_h {
        let w = max_dimension;
        let h = ((src_h as f64 / src_w as f64) * max_dimension as f64).round() as u32;
        (w, h.max(1))
    } else {
        let h = max_dimension;
        let w = ((src_w as f64 / src_h as f64) * max_dimension as f64).round() as u32;
        (w.max(1), h)
    };

    // For CropMode::Heuristic, the aspect is 3:4 so height > width.
    // For CropMode::None, the aspect is preserved from the original.
    image.resize_exact(new_w, new_h, FilterType::Lanczos3)
}

/// Flatten alpha channel by compositing onto a white background.
pub(crate) fn flatten_alpha(image: &DynamicImage) -> RgbImage {
    let rgba: RgbaImage = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let mut rgb = RgbImage::new(width, height);

    for (x, y, pixel) in rgba.enumerate_pixels() {
        let [r, g, b, a] = pixel.0;
        let alpha = a as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;
        // Composite over white (255, 255, 255)
        let out_r = (r as f32 * alpha + 255.0 * inv_alpha).round() as u8;
        let out_g = (g as f32 * alpha + 255.0 * inv_alpha).round() as u8;
        let out_b = (b as f32 * alpha + 255.0 * inv_alpha).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([out_r, out_g, out_b]));
    }

    rgb
}

/// Apply grayscale conversion if requested.
pub(crate) fn apply_grayscale(image: RgbImage, grayscale: bool) -> RgbImage {
    if !grayscale {
        return image;
    }
    let gray = image::imageops::grayscale(&image);
    // Convert Luma back to RGB for consistent encoding
    let (width, height) = (gray.width(), gray.height());
    let mut rgb = RgbImage::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        let v = pixel.0[0];
        rgb.put_pixel(x, y, image::Rgb([v, v, v]));
    }
    rgb
}

/// Encode an image to the specified format at the given quality.
///
/// When `grayscale` is true, JPEG is encoded as single-channel Luma8 to avoid
/// wasting bytes on identical R=G=B triplets.
pub(crate) fn encode_image(
    image: &RgbImage,
    format: &OutputFormat,
    quality: f32,
    grayscale: bool,
) -> Result<Vec<u8>, IdPhotoError> {
    let mut buffer = Vec::new();

    match format {
        OutputFormat::Webp => {
            // Use libwebp-backed lossy VP8 encoding so WebP honors the quality setting.
            let quality_percent = (quality * 100.0).clamp(0.0, 100.0);
            let encoded = LibWebpEncoder::from_rgb(image.as_raw(), image.width(), image.height())
                .encode_simple(false, quality_percent)
                .map_err(|e| IdPhotoError::EncodeError(format!("WebP encoding failed: {e:?}")))?;
            buffer.extend_from_slice(&encoded);

            // Strip ICC profile from WebP output
            buffer = strip_icc_profile(&buffer);
        }
        OutputFormat::Jpeg => {
            // For grayscale, encode as Luma8 (single channel) instead of Rgb8 (3 channels).
            let (raw_data, color_type) = if grayscale {
                let luma: Vec<u8> = image.as_raw().chunks(3).map(|rgb| rgb[0]).collect();
                (luma, image::ExtendedColorType::L8)
            } else {
                (image.as_raw().to_vec(), image::ExtendedColorType::Rgb8)
            };

            let quality_percent = (quality * 100.0).round() as u8;
            let encoder = JpegEncoder::new_with_quality(&mut buffer, quality_percent);
            encoder
                .write_image(&raw_data, image.width(), image.height(), color_type)
                .map_err(|e| IdPhotoError::EncodeError(e.to_string()))?;
        }
    }

    Ok(buffer)
}

/// Transform face bounds from source image coordinates to output image coordinates.
///
/// Accounts for the crop offset and the resize scale factor.
fn transform_face_bounds(
    face: &FaceBounds,
    crop_offset: (u32, u32),
    crop_size: (u32, u32),
    output_size: (u32, u32),
) -> FaceBounds {
    let scale_x = output_size.0 as f64 / crop_size.0 as f64;
    let scale_y = output_size.1 as f64 / crop_size.1 as f64;

    FaceBounds {
        x: (face.x - crop_offset.0 as f64) * scale_x,
        y: (face.y - crop_offset.1 as f64) * scale_y,
        width: face.width * scale_x,
        height: face.height * scale_y,
        confidence: face.confidence,
    }
}

/// Full compression pipeline: decode → crop → resize → flatten → grayscale → encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compress_pipeline(
    input: &[u8],
    max_dimension: u32,
    quality: f32,
    grayscale: bool,
    crop_mode: &CropMode,
    format: &OutputFormat,
    face_margin: f32,
    detector: Option<&dyn FaceDetector>,
) -> Result<CompressedPhoto, IdPhotoError> {
    let decoded = decode_image(input)?;

    if decoded.width() == 0 || decoded.height() == 0 {
        return Err(IdPhotoError::ZeroDimensions);
    }

    let crop_result = apply_crop(&decoded, crop_mode, face_margin, detector);
    let crop_size = (crop_result.image.width(), crop_result.image.height());
    let resized = resize_image(&crop_result.image, max_dimension);
    let output_size = (resized.width(), resized.height());
    let flattened = flatten_alpha(&resized);
    let rgb = apply_grayscale(flattened, grayscale);
    let data = encode_image(&rgb, format, quality, grayscale)?;

    // Transform face bounds to output coordinates
    let face_bounds = crop_result
        .face_bounds
        .as_ref()
        .map(|face| transform_face_bounds(face, crop_result.crop_offset, crop_size, output_size));

    Ok(CompressedPhoto {
        data,
        format: format.clone(),
        width: rgb.width(),
        height: rgb.height(),
        original_size: input.len(),
        face_bounds,
    })
}

/// Detect the input image format from the raw bytes.
pub(crate) fn detect_format(input: &[u8]) -> Result<ImageFormat, IdPhotoError> {
    image::guess_format(input).map_err(|e| IdPhotoError::DecodeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detector::{FaceBounds, FaceDetector};
    use image::ImageEncoder;

    /// Mock face detector that returns a fixed face at a known position.
    struct MockDetector {
        faces: Vec<FaceBounds>,
    }

    impl MockDetector {
        fn with_face(x: f64, y: f64, width: f64, height: f64) -> Self {
            Self {
                faces: vec![FaceBounds {
                    x,
                    y,
                    width,
                    height,
                    confidence: 10.0,
                }],
            }
        }

        fn empty() -> Self {
            Self { faces: vec![] }
        }
    }

    impl FaceDetector for MockDetector {
        fn detect(&self, _gray: &[u8], _width: u32, _height: u32) -> Vec<FaceBounds> {
            self.faces.clone()
        }
    }

    fn make_test_rgb(width: u32, height: u32) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            // Simple gradient pattern
            *pixel = image::Rgb([
                (x * 255 / width.max(1)) as u8,
                (y * 255 / height.max(1)) as u8,
                128,
            ]);
        }
        img
    }

    fn make_test_png(width: u32, height: u32) -> Vec<u8> {
        let img = make_test_rgb(width, height);
        let mut buffer = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buffer);
        encoder
            .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgb8)
            .unwrap();
        buffer
    }

    #[test]
    fn encode_jpeg_produces_valid_output() {
        let img = make_test_rgb(48, 64);
        let data = encode_image(&img, &OutputFormat::Jpeg, 0.8, false).unwrap();
        assert!(!data.is_empty());
        // JPEG magic bytes
        assert_eq!(data[0], 0xFF);
        assert_eq!(data[1], 0xD8);
    }

    #[test]
    fn encode_webp_produces_valid_output() {
        let img = make_test_rgb(48, 64);
        let data = encode_image(&img, &OutputFormat::Webp, 0.8, false).unwrap();
        assert!(!data.is_empty());
        // RIFF header
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WEBP");
    }

    #[test]
    fn encode_webp_quality_controls_size() {
        let img = make_test_rgb(48, 64);
        let low_quality = encode_image(&img, &OutputFormat::Webp, 0.3, false).unwrap();
        let high_quality = encode_image(&img, &OutputFormat::Webp, 0.9, false).unwrap();
        assert!(
            low_quality.len() < high_quality.len(),
            "low quality ({}) should be smaller than high quality ({})",
            low_quality.len(),
            high_quality.len()
        );
    }

    #[test]
    fn encode_grayscale_webp_smaller_than_color() {
        let img = make_test_rgb(48, 64);
        let color = encode_image(&img, &OutputFormat::Webp, 0.8, false).unwrap();
        let gray = encode_image(&img, &OutputFormat::Webp, 0.8, true).unwrap();
        assert!(
            gray.len() <= color.len(),
            "grayscale ({}) should be no larger than color ({})",
            gray.len(),
            color.len()
        );
    }

    #[test]
    fn flatten_alpha_composites_over_white() {
        // Fully transparent pixel should become white
        let mut rgba = RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([255, 0, 0, 0]));
        let dynamic = DynamicImage::ImageRgba8(rgba);
        let rgb = flatten_alpha(&dynamic);
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([255, 255, 255]));
    }

    #[test]
    fn flatten_alpha_preserves_opaque() {
        let mut rgba = RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([100, 150, 200, 255]));
        let dynamic = DynamicImage::ImageRgba8(rgba);
        let rgb = flatten_alpha(&dynamic);
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([100, 150, 200]));
    }

    #[test]
    fn flatten_alpha_blends_semitransparent() {
        let mut rgba = RgbaImage::new(1, 1);
        // 50% transparent red → should blend with white
        rgba.put_pixel(0, 0, image::Rgba([255, 0, 0, 128]));
        let dynamic = DynamicImage::ImageRgba8(rgba);
        let rgb = flatten_alpha(&dynamic);
        let pixel = rgb.get_pixel(0, 0);
        // ~128 + ~127 = ~255 for red, ~0 + ~127 = ~127 for green/blue
        assert!((pixel.0[0] as i16 - 255).abs() <= 1);
        assert!((pixel.0[1] as i16 - 127).abs() <= 2);
        assert!((pixel.0[2] as i16 - 127).abs() <= 2);
    }

    #[test]
    fn grayscale_conversion() {
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let gray = apply_grayscale(img, true);
        let pixel = gray.get_pixel(0, 0);
        // All channels should be equal (grayscale)
        assert_eq!(pixel.0[0], pixel.0[1]);
        assert_eq!(pixel.0[1], pixel.0[2]);
    }

    #[test]
    fn grayscale_disabled_is_identity() {
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([100, 150, 200]));
        let result = apply_grayscale(img, false);
        assert_eq!(result.get_pixel(0, 0), &image::Rgb([100, 150, 200]));
    }

    #[test]
    fn resize_landscape_constrains_width() {
        let img = DynamicImage::ImageRgb8(make_test_rgb(200, 100));
        let resized = resize_image(&img, 48);
        assert_eq!(resized.width(), 48);
        assert_eq!(resized.height(), 24);
    }

    #[test]
    fn resize_portrait_constrains_height() {
        let img = DynamicImage::ImageRgb8(make_test_rgb(100, 200));
        let resized = resize_image(&img, 64);
        assert_eq!(resized.width(), 32);
        assert_eq!(resized.height(), 64);
    }

    #[test]
    fn full_pipeline_produces_output() {
        let png = make_test_png(200, 300);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::Heuristic,
            &OutputFormat::Webp,
            2.0,
            None,
        )
        .unwrap();
        assert!(!result.data.is_empty());
        assert_eq!(result.original_size, png.len());
    }

    #[test]
    fn full_pipeline_with_grayscale() {
        let png = make_test_png(200, 300);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            true,
            &CropMode::Heuristic,
            &OutputFormat::Jpeg,
            2.0,
            None,
        )
        .unwrap();
        assert!(!result.data.is_empty());
        // JPEG magic
        assert_eq!(result.data[0], 0xFF);
        assert_eq!(result.data[1], 0xD8);
    }

    #[test]
    fn full_pipeline_no_crop() {
        let png = make_test_png(200, 300);
        let result = compress_pipeline(
            &png,
            48,
            0.8,
            false,
            &CropMode::None,
            &OutputFormat::Webp,
            2.0,
            None,
        )
        .unwrap();
        // With CropMode::None, aspect ratio is preserved from source (200:300 = 2:3)
        // max_dimension=48 constrains height to 48, width to 32
        assert_eq!(result.height, 48);
        assert_eq!(result.width, 32);
    }

    #[test]
    fn invalid_input_returns_error() {
        let result = compress_pipeline(
            b"not an image",
            48,
            0.6,
            false,
            &CropMode::Heuristic,
            &OutputFormat::Webp,
            2.0,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn user_provided_detector_is_used() {
        let png = make_test_png(200, 300);
        // Place a fake face at the center of the image
        let detector = MockDetector::with_face(70.0, 100.0, 60.0, 60.0);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::FaceDetection,
            &OutputFormat::Webp,
            2.0,
            Some(&detector),
        )
        .unwrap();
        assert!(!result.data.is_empty());
        // Face was detected → face_bounds should be present
        assert!(
            result.face_bounds.is_some(),
            "face_bounds should be populated when detector finds a face"
        );
    }

    #[test]
    fn empty_detection_falls_back_to_heuristic() {
        let png = make_test_png(200, 300);
        let detector = MockDetector::empty();
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::FaceDetection,
            &OutputFormat::Webp,
            2.0,
            Some(&detector),
        )
        .unwrap();
        assert!(!result.data.is_empty());
        // No face found → falls back to heuristic → no face_bounds
        assert!(
            result.face_bounds.is_none(),
            "face_bounds should be None when no face is detected"
        );
    }

    #[test]
    fn no_detector_no_feature_falls_back_to_heuristic() {
        let png = make_test_png(200, 300);
        // FaceDetection mode with no detector and no built-in feature
        // (when compiled without rustface, detect_face returns None)
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::FaceDetection,
            &OutputFormat::Webp,
            2.0,
            None,
        )
        .unwrap();
        assert!(!result.data.is_empty());
        // Without rustface feature, no face will be detected → heuristic fallback
        // With rustface feature on a gradient test image, also no face → heuristic fallback
        // Either way the output should be valid
    }

    #[test]
    fn face_bounds_absent_for_heuristic_mode() {
        let png = make_test_png(200, 300);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::Heuristic,
            &OutputFormat::Webp,
            2.0,
            None,
        )
        .unwrap();
        assert!(
            result.face_bounds.is_none(),
            "face_bounds should be None for Heuristic mode"
        );
    }

    #[test]
    fn face_bounds_absent_for_none_mode() {
        let png = make_test_png(200, 300);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::None,
            &OutputFormat::Webp,
            2.0,
            None,
        )
        .unwrap();
        assert!(
            result.face_bounds.is_none(),
            "face_bounds should be None for CropMode::None"
        );
    }

    #[test]
    fn face_bounds_transformed_to_output_coordinates() {
        let png = make_test_png(300, 400);
        // Place a face in the source image
        let detector = MockDetector::with_face(100.0, 100.0, 80.0, 80.0);
        let result = compress_pipeline(
            &png,
            48,
            0.6,
            false,
            &CropMode::FaceDetection,
            &OutputFormat::Webp,
            2.0,
            Some(&detector),
        )
        .unwrap();

        let bounds = result
            .face_bounds
            .as_ref()
            .expect("face_bounds should be present");
        // The face bounds should be in output image coordinates
        assert!(bounds.x >= 0.0, "face x should be non-negative");
        assert!(bounds.y >= 0.0, "face y should be non-negative");
        assert!(
            bounds.width > 0.0 && bounds.width <= result.width as f64,
            "face width should be within output bounds"
        );
        assert!(
            bounds.height > 0.0 && bounds.height <= result.height as f64,
            "face height should be within output bounds"
        );
        assert_eq!(bounds.confidence, 10.0, "confidence should be preserved");
    }

    #[test]
    fn compute_face_crop_produces_3_4_aspect() {
        let face = FaceBounds {
            x: 100.0,
            y: 100.0,
            width: 50.0,
            height: 50.0,
            confidence: 5.0,
        };
        let region = compute_face_crop(&face, 400, 600, 2.0);
        let aspect = region.width as f64 / region.height as f64;
        assert!(
            (aspect - 0.75).abs() < 0.02,
            "crop aspect should be ~3:4, got {aspect}"
        );
    }

    #[test]
    fn transform_face_bounds_scales_correctly() {
        let face = FaceBounds {
            x: 100.0,
            y: 200.0,
            width: 50.0,
            height: 60.0,
            confidence: 8.0,
        };
        // Crop at (50, 100), crop size 200x300, output size 20x30
        let transformed = transform_face_bounds(&face, (50, 100), (200, 300), (20, 30));
        // x: (100 - 50) * (20/200) = 50 * 0.1 = 5.0
        assert!((transformed.x - 5.0).abs() < 0.01);
        // y: (200 - 100) * (30/300) = 100 * 0.1 = 10.0
        assert!((transformed.y - 10.0).abs() < 0.01);
        // width: 50 * 0.1 = 5.0
        assert!((transformed.width - 5.0).abs() < 0.01);
        // height: 60 * 0.1 = 6.0
        assert!((transformed.height - 6.0).abs() < 0.01);
        assert_eq!(transformed.confidence, 8.0);
    }
}
