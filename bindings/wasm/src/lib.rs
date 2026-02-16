use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Options for photo compression, passed as a JavaScript object.
///
/// All fields are optional. When a `preset` is specified, its defaults apply
/// and individual fields override them.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct CompressOptions {
    pub preset: Option<String>,
    pub max_dimension: Option<u32>,
    pub quality: Option<f32>,
    pub grayscale: Option<bool>,
    pub crop_mode: Option<String>,
    pub format: Option<String>,
    pub face_margin: Option<f32>,
}

fn format_to_str(format: &idphoto::OutputFormat) -> &'static str {
    match format {
        idphoto::OutputFormat::Webp => "webp",
        idphoto::OutputFormat::Jpeg => "jpeg",
    }
}

fn string_to_preset(preset: &str) -> Result<idphoto::Preset, JsValue> {
    match preset {
        "qr-code" => Ok(idphoto::Preset::QrCode),
        "qr-code-match" => Ok(idphoto::Preset::QrCodeMatch),
        "print" => Ok(idphoto::Preset::Print),
        "display" => Ok(idphoto::Preset::Display),
        _ => Err(make_error(
            "INVALID_OPTIONS",
            &format!("unknown preset: {preset}"),
        )),
    }
}

fn string_to_crop_mode(mode: &str) -> Result<idphoto::CropMode, JsValue> {
    match mode {
        "heuristic" => Ok(idphoto::CropMode::Heuristic),
        "none" => Ok(idphoto::CropMode::None),
        "face-detection" => Ok(idphoto::CropMode::FaceDetection),
        _ => Err(make_error(
            "INVALID_OPTIONS",
            &format!("unknown crop mode: {mode}"),
        )),
    }
}

fn string_to_format(format: &str) -> Result<idphoto::OutputFormat, JsValue> {
    match format {
        "webp" => Ok(idphoto::OutputFormat::Webp),
        "jpeg" => Ok(idphoto::OutputFormat::Jpeg),
        _ => Err(make_error(
            "INVALID_OPTIONS",
            &format!("unknown format: {format}"),
        )),
    }
}

/// Create a JS `Error` with a `code` property.
fn make_error(code: &str, message: &str) -> JsValue {
    let err = js_sys::Error::new(message);
    let _ = js_sys::Reflect::set(&err, &"code".into(), &JsValue::from_str(code));
    JsValue::from(err)
}

/// Convert an `IdPhotoError` into a JS `Error` with a machine-readable `code` property.
fn to_js_error(e: idphoto::IdPhotoError) -> JsValue {
    let (code, message) = match &e {
        idphoto::IdPhotoError::DecodeError(_) => ("DECODE_ERROR", e.to_string()),
        idphoto::IdPhotoError::UnsupportedFormat => ("UNSUPPORTED_FORMAT", e.to_string()),
        idphoto::IdPhotoError::ZeroDimensions => ("ZERO_DIMENSIONS", e.to_string()),
        idphoto::IdPhotoError::EncodeError(_) => ("ENCODE_ERROR", e.to_string()),
        idphoto::IdPhotoError::InvalidQuality(_) => ("INVALID_QUALITY", e.to_string()),
        idphoto::IdPhotoError::InvalidMaxDimension => ("INVALID_MAX_DIMENSION", e.to_string()),
    };
    make_error(code, &message)
}

fn parse_options(options: JsValue) -> Result<CompressOptions, JsValue> {
    if options.is_undefined() || options.is_null() {
        Ok(CompressOptions::default())
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| make_error("INVALID_OPTIONS", &format!("invalid options: {e}")))
    }
}

/// Apply parsed `CompressOptions` to a `PhotoCompressor`, returning the
/// configured compressor ready for compression.
fn apply_options(
    mut compressor: idphoto::PhotoCompressor,
    opts: &CompressOptions,
) -> Result<idphoto::PhotoCompressor, JsValue> {
    if let Some(ref p) = opts.preset {
        compressor = compressor.preset(string_to_preset(p)?);
    }
    if let Some(dim) = opts.max_dimension {
        compressor = compressor.max_dimension(dim);
    }
    if let Some(q) = opts.quality {
        compressor = compressor.quality(q);
    }
    if let Some(g) = opts.grayscale {
        compressor = compressor.grayscale(g);
    }
    if let Some(ref mode) = opts.crop_mode {
        compressor = compressor.crop_mode(string_to_crop_mode(mode)?);
    }
    if let Some(ref fmt) = opts.format {
        compressor = compressor.format(string_to_format(fmt)?);
    }
    if let Some(margin) = opts.face_margin {
        compressor = compressor.face_margin(margin);
    }
    Ok(compressor)
}

/// Build a plain JS object from a `CompressedPhoto`.
fn build_photo_object(photo: &idphoto::CompressedPhoto) -> Result<JsValue, JsValue> {
    let obj = js_sys::Object::new();
    let data = js_sys::Uint8Array::from(&photo.data[..]);
    js_sys::Reflect::set(&obj, &"data".into(), &data)?;
    js_sys::Reflect::set(
        &obj,
        &"format".into(),
        &JsValue::from_str(format_to_str(&photo.format)),
    )?;
    js_sys::Reflect::set(&obj, &"width".into(), &JsValue::from(photo.width))?;
    js_sys::Reflect::set(&obj, &"height".into(), &JsValue::from(photo.height))?;
    js_sys::Reflect::set(
        &obj,
        &"originalSize".into(),
        &JsValue::from(photo.original_size as u32),
    )?;

    let fb = match photo.face_bounds.as_ref() {
        Some(bounds) => {
            let fb_obj = js_sys::Object::new();
            js_sys::Reflect::set(&fb_obj, &"x".into(), &JsValue::from(bounds.x))?;
            js_sys::Reflect::set(&fb_obj, &"y".into(), &JsValue::from(bounds.y))?;
            js_sys::Reflect::set(&fb_obj, &"width".into(), &JsValue::from(bounds.width))?;
            js_sys::Reflect::set(&fb_obj, &"height".into(), &JsValue::from(bounds.height))?;
            js_sys::Reflect::set(
                &fb_obj,
                &"confidence".into(),
                &JsValue::from(bounds.confidence),
            )?;
            JsValue::from(fb_obj)
        }
        None => JsValue::NULL,
    };
    js_sys::Reflect::set(&obj, &"faceBounds".into(), &fb)?;

    Ok(JsValue::from(obj))
}

/// Compress an identity photo with the given options.
///
/// @param input - Raw image bytes (JPEG, PNG, or WebP)
/// @param options - Optional object with fields: preset, maxDimension,
///   quality, grayscale, cropMode, format, faceMargin
#[wasm_bindgen]
pub fn compress(input: Vec<u8>, options: JsValue) -> Result<JsValue, JsValue> {
    let opts = parse_options(options)?;

    let compressor = idphoto::PhotoCompressor::new(input).map_err(to_js_error)?;
    let compressor = apply_options(compressor, &opts)?;

    let result = compressor.compress().map_err(to_js_error)?;

    build_photo_object(&result)
}

/// Compress an identity photo to fit within a byte budget.
///
/// Uses binary search over quality (8 iterations) to find the highest quality
/// that produces output within `max_bytes`.
///
/// @param input - Raw image bytes (JPEG, PNG, or WebP)
/// @param max_bytes - Maximum output size in bytes
/// @param options - Optional object with fields: preset, maxDimension,
///   grayscale, cropMode, format, faceMargin
#[wasm_bindgen(js_name = "compressToFit")]
pub fn compress_to_fit(
    input: Vec<u8>,
    max_bytes: usize,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let opts = parse_options(options)?;

    let compressor = idphoto::PhotoCompressor::new(input).map_err(to_js_error)?;
    let compressor = apply_options(compressor, &opts)?;

    let result = compressor.compress_to_fit(max_bytes).map_err(to_js_error)?;

    let photo_obj = build_photo_object(&result.photo)?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &"photo".into(), &photo_obj)?;
    js_sys::Reflect::set(
        &obj,
        &"qualityUsed".into(),
        &JsValue::from(result.quality_used),
    )?;
    js_sys::Reflect::set(
        &obj,
        &"reachedTarget".into(),
        &JsValue::from(result.reached_target),
    )?;

    Ok(JsValue::from(obj))
}
