use idphoto_wasm::{compress, compress_to_fit};
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, Rgb, RgbImage};
use wasm_bindgen::JsValue;
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

/// Helper: build a JS options object from key-value pairs.
/// Uses js_sys to construct a real JS object (avoids serde_json roundtrip quirks).
fn js_options(entries: &[(&str, JsValue)]) -> JsValue {
    let obj = js_sys::Object::new();
    for (key, val) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(key), val).unwrap();
    }
    obj.into()
}

/// Helper: extract a property from a plain JS object by key.
fn get_prop(obj: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

#[wasm_bindgen_test]
fn basic_compress_with_defaults() {
    let png = make_test_png(200, 300);
    let result = compress(png.clone(), JsValue::UNDEFINED).unwrap();

    let data = js_sys::Uint8Array::from(get_prop(&result, "data"));
    assert!(data.length() > 0);
    let width = get_prop(&result, "width").as_f64().unwrap() as u32;
    let height = get_prop(&result, "height").as_f64().unwrap() as u32;
    assert!(width <= 48);
    assert!(height <= 64);
    let original_size = get_prop(&result, "originalSize").as_f64().unwrap() as usize;
    assert_eq!(original_size, png.len());
}

#[wasm_bindgen_test]
fn compress_with_null_options() {
    let png = make_test_png(200, 300);
    let result = compress(png, JsValue::NULL).unwrap();

    let data = js_sys::Uint8Array::from(get_prop(&result, "data"));
    assert!(data.length() > 0);
}

#[wasm_bindgen_test]
fn compress_to_fit_respects_byte_budget() {
    let png = make_test_png(200, 300);
    let result = compress_to_fit(png, 10_000, JsValue::UNDEFINED).unwrap();

    let reached_target = get_prop(&result, "reachedTarget").as_bool().unwrap();
    assert!(reached_target);

    let photo = get_prop(&result, "photo");
    let data = js_sys::Uint8Array::from(get_prop(&photo, "data"));
    assert!(data.length() <= 10_000);

    let quality_used = get_prop(&result, "qualityUsed").as_f64().unwrap() as f32;
    assert!(quality_used > 0.0);
}

#[wasm_bindgen_test]
fn all_presets_produce_valid_output() {
    let png = make_test_png(200, 300);

    for preset in ["qr-code", "qr-code-match", "print", "display"] {
        let opts = js_options(&[("preset", JsValue::from_str(preset))]);
        let result = compress(png.clone(), opts).unwrap();

        let data = js_sys::Uint8Array::from(get_prop(&result, "data"));
        assert!(
            data.length() > 0,
            "preset {preset} produced empty output"
        );
        let width = get_prop(&result, "width").as_f64().unwrap() as u32;
        let height = get_prop(&result, "height").as_f64().unwrap() as u32;
        assert!(width > 0, "preset {preset} produced zero width");
        assert!(height > 0, "preset {preset} produced zero height");
    }
}

#[wasm_bindgen_test]
fn compress_with_specific_options() {
    let png = make_test_png(200, 300);
    let opts = js_options(&[
        ("maxDimension", JsValue::from(100)),
        ("quality", JsValue::from(0.8)),
        ("grayscale", JsValue::from(true)),
        ("format", JsValue::from_str("jpeg")),
    ]);
    let result = compress(png, opts).unwrap();

    let data = js_sys::Uint8Array::from(get_prop(&result, "data"));
    assert!(data.length() > 0);
    let width = get_prop(&result, "width").as_f64().unwrap() as u32;
    let height = get_prop(&result, "height").as_f64().unwrap() as u32;
    assert!(width <= 100);
    assert!(height <= 100);
    let format = get_prop(&result, "format").as_string().unwrap();
    assert_eq!(format, "jpeg");
    // JPEG magic bytes
    let first_two = data.slice(0, 2).to_vec();
    assert_eq!(&first_two, &[0xFF, 0xD8]);
}

#[wasm_bindgen_test]
fn invalid_input_returns_error() {
    let result = compress(b"not an image".to_vec(), JsValue::UNDEFINED);

    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn invalid_options_returns_error() {
    let png = make_test_png(200, 300);
    let bad_options = JsValue::from_str("not an object");
    let result = compress(png, bad_options);

    assert!(result.is_err());
}
