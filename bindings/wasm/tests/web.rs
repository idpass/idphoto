use idphoto_wasm::{compress, compress_to_fit};
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, Rgb, RgbImage};
use wasm_bindgen_test::*;

fn make_test_png(width: u32, height: u32) -> Vec<u8> {
    let mut img = RgbImage::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([
            (x * 255 / width.max(1)) as u8,
            (y * 255 / height.max(1)) as u8,
            128,
        ]);
    }

    let mut buffer = Vec::new();
    PngEncoder::new(&mut buffer)
        .write_image(img.as_raw(), width, height, ExtendedColorType::Rgb8)
        .unwrap();
    buffer
}

#[wasm_bindgen_test]
fn basic_compress_with_synthetic_png() {
    let png = make_test_png(200, 300);
    let result = compress(png.clone(), None, None, None, None, None, None, None).unwrap();

    assert!(!result.data().is_empty());
    assert!(result.width() <= 48);
    assert!(result.height() <= 64);
    assert_eq!(result.original_size(), png.len());
}

#[wasm_bindgen_test]
fn compress_to_fit_respects_byte_budget() {
    let png = make_test_png(200, 300);
    let result = compress_to_fit(png, 10_000, None, None, None, None, None, None).unwrap();
    let photo = result.photo();

    assert!(result.reached_target());
    assert!(photo.data().len() <= 10_000);
    assert!(result.quality_used() > 0.0);
}

#[wasm_bindgen_test]
fn all_presets_produce_valid_output() {
    let png = make_test_png(200, 300);

    for preset in ["qr-code", "qr-match", "print", "display"] {
        let result = compress(
            png.clone(),
            Some(preset.to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(
            !result.data().is_empty(),
            "preset {preset} produced empty output"
        );
        assert!(result.width() > 0, "preset {preset} produced zero width");
        assert!(result.height() > 0, "preset {preset} produced zero height");
    }
}

#[wasm_bindgen_test]
fn invalid_input_returns_error() {
    let result = compress(
        b"not an image".to_vec(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    assert!(result.is_err());
}
