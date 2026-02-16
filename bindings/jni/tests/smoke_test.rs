use idphoto_jni::*;

fn make_test_png(width: u32, height: u32) -> Vec<u8> {
    use image::codecs::png::PngEncoder;
    use image::{ImageEncoder, RgbImage};

    let mut img = RgbImage::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb([
            (x * 255 / width.max(1)) as u8,
            (y * 255 / height.max(1)) as u8,
            128,
        ]);
    }
    let mut buffer = Vec::new();
    let encoder = PngEncoder::new(&mut buffer);
    encoder
        .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgb8)
        .unwrap();
    buffer
}

#[test]
fn default_options_factory_works() {
    let opts = default_compress_options();
    assert!(opts.preset.is_none());
    assert!(opts.max_dimension.is_none());
    assert!(opts.quality.is_none());
    assert!(opts.grayscale.is_none());
    assert!(opts.crop_mode.is_none());
    assert!(opts.format.is_none());
    assert!(opts.face_margin.is_none());
}

#[test]
fn compress_with_preset_works() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        preset: Some(Preset::QrCode),
        ..default_compress_options()
    };
    let result = compress(png, options).unwrap();
    assert!(!result.data.is_empty());
    assert!(result.width <= 48);
    assert!(result.height <= 64);
}

#[test]
fn compress_with_all_parameters() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        max_dimension: Some(48),
        quality: Some(0.8),
        grayscale: Some(false),
        crop_mode: Some(CropMode::Heuristic),
        format: Some(OutputFormat::Jpeg),
        face_margin: Some(2.0),
        ..default_compress_options()
    };
    let result = compress(png, options).unwrap();
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0], 0xFF);
    assert_eq!(result.data[1], 0xD8);
}

#[test]
fn compress_to_fit_with_preset_works() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        preset: Some(Preset::QrCode),
        ..default_compress_options()
    };
    let result = compress_to_fit(png, 10_000, options).unwrap();
    assert!(result.reached_target);
    assert!(result.photo.data.len() <= 10_000);
}

#[test]
fn compress_to_fit_with_all_parameters() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        max_dimension: Some(48),
        grayscale: Some(true),
        crop_mode: Some(CropMode::Heuristic),
        format: Some(OutputFormat::Jpeg),
        face_margin: Some(2.0),
        ..default_compress_options()
    };
    let result = compress_to_fit(png, 10_000, options).unwrap();
    assert!(result.reached_target);
    assert!(result.quality_used > 0.0);
}

#[test]
fn face_detection_crop_mode_works() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        max_dimension: Some(48),
        quality: Some(0.8),
        grayscale: Some(false),
        crop_mode: Some(CropMode::FaceDetection),
        format: Some(OutputFormat::Webp),
        face_margin: Some(2.0),
        ..default_compress_options()
    };
    let result = compress(png, options).unwrap();
    assert!(!result.data.is_empty());
}

#[test]
fn invalid_input_returns_error() {
    let result = compress(b"not an image".to_vec(), default_compress_options());
    assert!(result.is_err());
}

#[test]
fn all_presets_work() {
    let png = make_test_png(200, 300);
    for preset in [
        Preset::QrCode,
        Preset::QrCodeMatch,
        Preset::Print,
        Preset::Display,
    ] {
        let options = CompressOptions {
            preset: Some(preset),
            ..default_compress_options()
        };
        let result = compress(png.clone(), options);
        assert!(result.is_ok());
    }
}

#[test]
fn invalid_quality_carries_value() {
    let png = make_test_png(200, 300);
    let options = CompressOptions {
        quality: Some(1.5),
        ..default_compress_options()
    };
    let result = compress(png, options);
    match result {
        Err(IdPhotoError::InvalidQuality { value }) => {
            assert!((value - 1.5).abs() < f32::EPSILON);
        }
        other => panic!("expected InvalidQuality, got {:?}", other),
    }
}

#[test]
fn preset_with_format_override() {
    let png = make_test_png(200, 300);
    // Start with QR preset (WebP, grayscale, 48px) then override format to JPEG
    let options = CompressOptions {
        preset: Some(Preset::QrCode),
        format: Some(OutputFormat::Jpeg),
        ..default_compress_options()
    };
    let result = compress(png, options).unwrap();
    // Should use JPEG despite QR preset defaulting to WebP
    assert_eq!(result.data[0], 0xFF);
    assert_eq!(result.data[1], 0xD8);
    // But keep other QR settings (48px max dimension)
    assert!(result.width <= 48);
}

#[test]
fn preset_with_dimension_override() {
    let png = make_test_png(200, 300);
    // Start with QR preset (48px) then override to 100px
    let options = CompressOptions {
        preset: Some(Preset::QrCode),
        max_dimension: Some(100),
        ..default_compress_options()
    };
    let result = compress(png, options).unwrap();
    // Should use the overridden max dimension
    assert!(result.width <= 100);
    assert!(result.height <= 134);
}

#[test]
fn default_options_use_library_defaults() {
    let png = make_test_png(200, 300);
    let result = compress(png, default_compress_options()).unwrap();
    assert!(!result.data.is_empty());
    // Library defaults: 48px max dimension
    assert!(result.width <= 48);
    assert!(result.height <= 64);
}

#[test]
fn compress_to_fit_with_default_options() {
    let png = make_test_png(200, 300);
    let result = compress_to_fit(png, 10_000, default_compress_options()).unwrap();
    assert!(result.reached_target);
    assert!(result.photo.data.len() <= 10_000);
}
