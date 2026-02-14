use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmFaceBounds {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    confidence: f64,
}

#[wasm_bindgen]
impl WasmFaceBounds {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f64 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
}

#[wasm_bindgen]
pub struct WasmCompressedPhoto {
    data: Vec<u8>,
    format: String,
    width: u32,
    height: u32,
    original_size: usize,
    face_bounds: Option<WasmFaceBounds>,
}

#[wasm_bindgen]
impl WasmCompressedPhoto {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        self.format.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter, js_name = "originalSize")]
    pub fn original_size(&self) -> usize {
        self.original_size
    }

    #[wasm_bindgen(getter, js_name = "faceBounds")]
    pub fn face_bounds(&self) -> Option<WasmFaceBounds> {
        self.face_bounds.as_ref().map(|fb| WasmFaceBounds {
            x: fb.x,
            y: fb.y,
            width: fb.width,
            height: fb.height,
            confidence: fb.confidence,
        })
    }
}

#[wasm_bindgen]
pub struct WasmFitResult {
    photo: WasmCompressedPhoto,
    quality_used: f32,
    reached_target: bool,
}

#[wasm_bindgen]
impl WasmFitResult {
    #[wasm_bindgen(getter)]
    pub fn photo(&self) -> WasmCompressedPhoto {
        WasmCompressedPhoto {
            data: self.photo.data.clone(),
            format: self.photo.format.clone(),
            width: self.photo.width,
            height: self.photo.height,
            original_size: self.photo.original_size,
            face_bounds: self.photo.face_bounds.as_ref().map(|fb| WasmFaceBounds {
                x: fb.x,
                y: fb.y,
                width: fb.width,
                height: fb.height,
                confidence: fb.confidence,
            }),
        }
    }

    #[wasm_bindgen(getter, js_name = "qualityUsed")]
    pub fn quality_used(&self) -> f32 {
        self.quality_used
    }

    #[wasm_bindgen(getter, js_name = "reachedTarget")]
    pub fn reached_target(&self) -> bool {
        self.reached_target
    }
}

fn format_to_string(format: &idphoto::OutputFormat) -> String {
    match format {
        idphoto::OutputFormat::Webp => "webp".to_string(),
        idphoto::OutputFormat::Jpeg => "jpeg".to_string(),
    }
}

fn string_to_preset(preset: &str) -> Result<idphoto::Preset, JsError> {
    match preset {
        "qr" | "qr-code" => Ok(idphoto::Preset::QrCode),
        "qr-match" | "qr-code-match" => Ok(idphoto::Preset::QrCodeMatch),
        "print" => Ok(idphoto::Preset::Print),
        "display" => Ok(idphoto::Preset::Display),
        _ => Err(JsError::new(&format!("unknown preset: {preset}"))),
    }
}

fn string_to_crop_mode(mode: &str) -> Result<idphoto::CropMode, JsError> {
    match mode {
        "heuristic" => Ok(idphoto::CropMode::Heuristic),
        "none" => Ok(idphoto::CropMode::None),
        "face-detection" => Ok(idphoto::CropMode::FaceDetection),
        _ => Err(JsError::new(&format!("unknown crop mode: {mode}"))),
    }
}

fn string_to_format(format: &str) -> Result<idphoto::OutputFormat, JsError> {
    match format {
        "webp" => Ok(idphoto::OutputFormat::Webp),
        "jpeg" => Ok(idphoto::OutputFormat::Jpeg),
        _ => Err(JsError::new(&format!("unknown format: {format}"))),
    }
}

fn face_bounds_to_wasm(bounds: &idphoto::FaceBounds) -> WasmFaceBounds {
    WasmFaceBounds {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
        confidence: bounds.confidence,
    }
}

/// Compress an identity photo with the given options.
///
/// @param input - Raw image bytes (JPEG, PNG, or WebP)
/// @param preset - "qr-code", "print", or "display" (optional, sets all defaults)
/// @param max_dimension - Maximum output dimension in pixels (overrides preset)
/// @param quality - Compression quality 0.0–1.0 (overrides preset)
/// @param grayscale - Convert to grayscale (overrides preset)
/// @param crop_mode - "heuristic", "none", or "face-detection" (overrides preset)
/// @param format - "webp" or "jpeg" (overrides preset)
/// @param face_margin - Face detection crop margin (overrides preset, default: 2.0)
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn compress(
    input: Vec<u8>,
    preset: Option<String>,
    max_dimension: Option<u32>,
    quality: Option<f32>,
    grayscale: Option<bool>,
    crop_mode: Option<String>,
    format: Option<String>,
    face_margin: Option<f32>,
) -> Result<WasmCompressedPhoto, JsError> {
    let mut compressor =
        idphoto::PhotoCompressor::new(input).map_err(|e| JsError::new(&e.to_string()))?;

    if let Some(p) = preset {
        compressor = compressor.preset(string_to_preset(&p)?);
    }
    if let Some(dim) = max_dimension {
        compressor = compressor.max_dimension(dim);
    }
    if let Some(q) = quality {
        compressor = compressor.quality(q);
    }
    if let Some(g) = grayscale {
        compressor = compressor.grayscale(g);
    }
    if let Some(mode) = crop_mode {
        compressor = compressor.crop_mode(string_to_crop_mode(&mode)?);
    }
    if let Some(fmt) = format {
        compressor = compressor.format(string_to_format(&fmt)?);
    }
    if let Some(margin) = face_margin {
        compressor = compressor.face_margin(margin);
    }

    let result = compressor
        .compress()
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(WasmCompressedPhoto {
        data: result.data,
        format: format_to_string(&result.format),
        width: result.width,
        height: result.height,
        original_size: result.original_size,
        face_bounds: result.face_bounds.as_ref().map(face_bounds_to_wasm),
    })
}

/// Compress an identity photo to fit within a byte budget.
///
/// Uses binary search over quality (8 iterations) to find the highest quality
/// that produces output within `max_bytes`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "compressToFit")]
pub fn compress_to_fit(
    input: Vec<u8>,
    max_bytes: usize,
    preset: Option<String>,
    max_dimension: Option<u32>,
    grayscale: Option<bool>,
    crop_mode: Option<String>,
    format: Option<String>,
    face_margin: Option<f32>,
) -> Result<WasmFitResult, JsError> {
    let mut compressor =
        idphoto::PhotoCompressor::new(input).map_err(|e| JsError::new(&e.to_string()))?;

    if let Some(p) = preset {
        compressor = compressor.preset(string_to_preset(&p)?);
    }
    if let Some(dim) = max_dimension {
        compressor = compressor.max_dimension(dim);
    }
    if let Some(g) = grayscale {
        compressor = compressor.grayscale(g);
    }
    if let Some(mode) = crop_mode {
        compressor = compressor.crop_mode(string_to_crop_mode(&mode)?);
    }
    if let Some(fmt) = format {
        compressor = compressor.format(string_to_format(&fmt)?);
    }
    if let Some(margin) = face_margin {
        compressor = compressor.face_margin(margin);
    }

    let result = compressor
        .compress_to_fit(max_bytes)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(WasmFitResult {
        photo: WasmCompressedPhoto {
            data: result.photo.data,
            format: format_to_string(&result.photo.format),
            width: result.photo.width,
            height: result.photo.height,
            original_size: result.photo.original_size,
            face_bounds: result.photo.face_bounds.as_ref().map(face_bounds_to_wasm),
        },
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}
