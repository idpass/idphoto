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
fn compress_with_preset_works() {
    let png = make_test_png(200, 300);
    let result = compress_with_preset(png, Preset::QrCode).unwrap();
    assert!(!result.data.is_empty());
    assert!(result.width <= 48);
    assert!(result.height <= 64);
}

#[test]
fn compress_with_all_parameters() {
    let png = make_test_png(200, 300);
    let result = compress(
        png,
        48,
        0.8,
        false,
        CropMode::Heuristic,
        OutputFormat::Jpeg,
        2.0,
    )
    .unwrap();
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0], 0xFF);
    assert_eq!(result.data[1], 0xD8);
}

#[test]
fn compress_to_fit_with_preset_works() {
    let png = make_test_png(200, 300);
    let result = compress_to_fit_with_preset(png, 10_000, Preset::QrCode).unwrap();
    assert!(result.reached_target);
    assert!(result.photo.data.len() <= 10_000);
}

#[test]
fn compress_to_fit_with_all_parameters() {
    let png = make_test_png(200, 300);
    let result = compress_to_fit(
        png,
        10_000,
        48,
        true,
        CropMode::Heuristic,
        OutputFormat::Jpeg,
        2.0,
    )
    .unwrap();
    assert!(result.reached_target);
    assert!(result.quality_used > 0.0);
}

#[test]
fn face_detection_crop_mode_works() {
    let png = make_test_png(200, 300);
    let result = compress(
        png,
        48,
        0.8,
        false,
        CropMode::FaceDetection,
        OutputFormat::Webp,
        2.0,
    )
    .unwrap();
    assert!(!result.data.is_empty());
}

#[test]
fn invalid_input_returns_error() {
    let result = compress_with_preset(b"not an image".to_vec(), Preset::QrCode);
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
        let result = compress_with_preset(png.clone(), preset);
        assert!(result.is_ok());
    }
}
