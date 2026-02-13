use image::codecs::jpeg::JpegEncoder;
use image::codecs::webp::WebPEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageEncoder, ImageFormat, RgbImage, RgbaImage};

use crate::crop::{portrait_crop, CropRegion};
use crate::error::IdPhotoError;
use crate::webp_strip::strip_icc_profile;
use crate::{CompressedPhoto, CropMode, OutputFormat};

/// Decode input bytes into a `DynamicImage`.
pub(crate) fn decode_image(input: &[u8]) -> Result<DynamicImage, IdPhotoError> {
    image::load_from_memory(input).map_err(|e| IdPhotoError::DecodeError(e.to_string()))
}

/// Apply the crop mode to the source image.
///
/// `face_margin` controls how tightly face detection crops around the face
/// (crop_height = face_height × face_margin). Ignored for non-face-detection modes.
pub(crate) fn apply_crop(image: &DynamicImage, mode: &CropMode, face_margin: f32) -> DynamicImage {
    match mode {
        CropMode::None => image.clone(),
        CropMode::Heuristic => {
            let CropRegion {
                x,
                y,
                width,
                height,
            } = portrait_crop(image.width(), image.height());
            image.crop_imm(x, y, width, height)
        }
        #[cfg(feature = "face-detection")]
        CropMode::FaceDetection => {
            // Face detection with fallback to heuristic
            match detect_face_crop(image, face_margin) {
                Some(region) => image.crop_imm(region.x, region.y, region.width, region.height),
                None => apply_crop(image, &CropMode::Heuristic, face_margin),
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
/// When `grayscale` is true, encodes as single-channel Luma8 to avoid
/// wasting bytes on identical R=G=B triplets.
pub(crate) fn encode_image(
    image: &RgbImage,
    format: &OutputFormat,
    quality: f32,
    grayscale: bool,
) -> Result<Vec<u8>, IdPhotoError> {
    let mut buffer = Vec::new();

    // For grayscale, encode as Luma8 (single channel) instead of Rgb8 (3 channels).
    // The RGB image already has R=G=B after apply_grayscale, so we extract the luma channel.
    let (raw_data, color_type) = if grayscale {
        let luma: Vec<u8> = image.as_raw().chunks(3).map(|rgb| rgb[0]).collect();
        (luma, image::ExtendedColorType::L8)
    } else {
        (image.as_raw().to_vec(), image::ExtendedColorType::Rgb8)
    };

    match format {
        OutputFormat::Webp => {
            // The pure-Rust image-webp crate only supports lossless encoding.
            // The quality parameter is ignored for WebP — use JPEG if you need
            // lossy quality-based size control (e.g. compress_to_fit).
            let encoder = WebPEncoder::new_lossless(&mut buffer);
            encoder
                .write_image(&raw_data, image.width(), image.height(), color_type)
                .map_err(|e| IdPhotoError::EncodeError(e.to_string()))?;

            // Strip ICC profile from WebP output
            buffer = strip_icc_profile(&buffer);
        }
        OutputFormat::Jpeg => {
            let quality_percent = (quality * 100.0).round() as u8;
            let encoder = JpegEncoder::new_with_quality(&mut buffer, quality_percent);
            encoder
                .write_image(&raw_data, image.width(), image.height(), color_type)
                .map_err(|e| IdPhotoError::EncodeError(e.to_string()))?;
        }
    }

    Ok(buffer)
}

/// Full compression pipeline: decode → crop → resize → flatten → grayscale → encode.
pub(crate) fn compress_pipeline(
    input: &[u8],
    max_dimension: u32,
    quality: f32,
    grayscale: bool,
    crop_mode: &CropMode,
    format: &OutputFormat,
    face_margin: f32,
) -> Result<CompressedPhoto, IdPhotoError> {
    let decoded = decode_image(input)?;

    if decoded.width() == 0 || decoded.height() == 0 {
        return Err(IdPhotoError::ZeroDimensions);
    }

    let cropped = apply_crop(&decoded, crop_mode, face_margin);
    let resized = resize_image(&cropped, max_dimension);
    let flattened = flatten_alpha(&resized);
    let rgb = apply_grayscale(flattened, grayscale);
    let data = encode_image(&rgb, format, quality, grayscale)?;

    Ok(CompressedPhoto {
        data,
        format: format.clone(),
        width: rgb.width(),
        height: rgb.height(),
        original_size: input.len(),
    })
}

/// Detect the input image format from the raw bytes.
pub(crate) fn detect_format(input: &[u8]) -> Result<ImageFormat, IdPhotoError> {
    image::guess_format(input).map_err(|e| IdPhotoError::DecodeError(e.to_string()))
}

#[cfg(feature = "face-detection")]
fn detect_face_crop(image: &DynamicImage, face_margin: f32) -> Option<CropRegion> {
    let gray = image::imageops::grayscale(image);
    let (img_w, img_h) = (gray.width(), gray.height());

    let model_data: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../model/seeta_fd_frontal_v1.0.bin"
    ));
    let model = rustface::read_model(std::io::Cursor::new(model_data)).ok()?;
    let mut detector = rustface::create_detector_with_model(model);
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let faces = detector.detect(&rustface::ImageData::new(gray.as_raw(), img_w, img_h));

    if faces.is_empty() {
        return None;
    }

    // Use the highest-scoring face
    let face = faces
        .iter()
        .max_by(|a: &&rustface::FaceInfo, b: &&rustface::FaceInfo| {
            a.score()
                .partial_cmp(&b.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    let bbox = face.bbox();
    let face_cx = bbox.x() as f64 + bbox.width() as f64 / 2.0;
    let face_cy = bbox.y() as f64 + bbox.height() as f64 / 2.0;
    let face_h = bbox.height() as f64;

    // Size the crop relative to the face.
    // face_margin controls framing: 2.0 = ID photo (face + hair + shoulders),
    // 1.3 = tight face crop for algorithmic matching.
    let portrait_aspect = 3.0 / 4.0;
    let desired_crop_h = (face_h * face_margin as f64).round();
    let desired_crop_w = (desired_crop_h * portrait_aspect).round();

    // Clamp to image bounds — the crop can't exceed the source dimensions
    let crop_h = (desired_crop_h as u32).min(img_h);
    let crop_w = (desired_crop_w as u32).min(img_w);

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
        .min((img_h.saturating_sub(crop_h)) as f64) as u32;
    let x = (face_cx - crop_w as f64 / 2.0)
        .round()
        .max(0.0)
        .min((img_w.saturating_sub(crop_w)) as f64) as u32;

    Some(CropRegion {
        x,
        y,
        width: crop_w,
        height: crop_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

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
    fn encode_grayscale_webp_smaller_than_color() {
        let img = make_test_rgb(48, 64);
        let color = encode_image(&img, &OutputFormat::Webp, 0.8, false).unwrap();
        let gray = encode_image(&img, &OutputFormat::Webp, 0.8, true).unwrap();
        assert!(
            gray.len() < color.len(),
            "grayscale ({}) should be smaller than color ({})",
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
        );
        assert!(result.is_err());
    }
}
