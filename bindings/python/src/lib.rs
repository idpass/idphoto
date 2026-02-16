use idphoto_core::{CropMode, IdPhotoError, OutputFormat, PhotoCompressor, Preset};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

// Exception hierarchy: all library errors inherit from IdPhotoError.
// Argument validation errors (bad preset/crop_mode/format strings) stay as ValueError.
create_exception!(idphoto, IdPhotoException, PyException);
create_exception!(idphoto, DecodeError, IdPhotoException);
create_exception!(idphoto, UnsupportedFormatError, IdPhotoException);
create_exception!(idphoto, ZeroDimensionsError, IdPhotoException);
create_exception!(idphoto, EncodeError, IdPhotoException);
create_exception!(idphoto, InvalidQualityError, IdPhotoException);
create_exception!(idphoto, InvalidMaxDimensionError, IdPhotoException);

fn to_py_err(e: IdPhotoError) -> PyErr {
    match e {
        IdPhotoError::DecodeError(msg) => {
            DecodeError::new_err(format!("failed to decode image: {msg}"))
        }
        IdPhotoError::UnsupportedFormat => {
            UnsupportedFormatError::new_err("unsupported image format")
        }
        IdPhotoError::ZeroDimensions => {
            ZeroDimensionsError::new_err("image dimensions are zero")
        }
        IdPhotoError::EncodeError(msg) => {
            EncodeError::new_err(format!("failed to encode image: {msg}"))
        }
        IdPhotoError::InvalidQuality(q) => {
            InvalidQualityError::new_err(format!("quality must be between 0.0 and 1.0, got {q}"))
        }
        IdPhotoError::InvalidMaxDimension => {
            InvalidMaxDimensionError::new_err("max dimension must be > 0")
        }
    }
}

fn format_to_string(format: &OutputFormat) -> &'static str {
    match format {
        OutputFormat::Webp => "webp",
        OutputFormat::Jpeg => "jpeg",
    }
}

fn string_to_preset(preset: &str) -> PyResult<Preset> {
    match preset {
        "qr-code" => Ok(Preset::QrCode),
        "qr-code-match" => Ok(Preset::QrCodeMatch),
        "print" => Ok(Preset::Print),
        "display" => Ok(Preset::Display),
        _ => Err(PyValueError::new_err(format!("unknown preset: {preset}"))),
    }
}

fn string_to_crop_mode(mode: &str) -> PyResult<CropMode> {
    match mode {
        "heuristic" => Ok(CropMode::Heuristic),
        "none" => Ok(CropMode::None),
        "face-detection" => Ok(CropMode::FaceDetection),
        _ => Err(PyValueError::new_err(format!("unknown crop mode: {mode}"))),
    }
}

fn string_to_format(format: &str) -> PyResult<OutputFormat> {
    match format {
        "webp" => Ok(OutputFormat::Webp),
        "jpeg" => Ok(OutputFormat::Jpeg),
        _ => Err(PyValueError::new_err(format!("unknown format: {format}"))),
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct FaceBounds {
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub width: f64,
    #[pyo3(get)]
    pub height: f64,
    #[pyo3(get)]
    pub confidence: f64,
}

#[pymethods]
impl FaceBounds {
    fn __repr__(&self) -> String {
        format!(
            "FaceBounds(x={}, y={}, width={}, height={}, confidence={:.2})",
            self.x, self.y, self.width, self.height, self.confidence
        )
    }
}

#[pyclass(frozen)]
pub struct CompressResult {
    #[pyo3(get)]
    pub data: Py<pyo3::types::PyBytes>,
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub original_size: usize,
    #[pyo3(get)]
    pub face_bounds: Option<FaceBounds>,
}

#[pymethods]
impl CompressResult {
    fn __repr__(&self, py: Python<'_>) -> String {
        let size = self.data.bind(py).as_bytes().len();
        format!(
            "CompressResult(format='{}', width={}, height={}, size={}, original_size={})",
            self.format, self.width, self.height, size, self.original_size
        )
    }
}

#[pyclass(frozen)]
pub struct FitResult {
    #[pyo3(get)]
    pub data: Py<pyo3::types::PyBytes>,
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub original_size: usize,
    #[pyo3(get)]
    pub face_bounds: Option<FaceBounds>,
    #[pyo3(get)]
    pub quality_used: f32,
    #[pyo3(get)]
    pub reached_target: bool,
}

#[pymethods]
impl FitResult {
    fn __repr__(&self, py: Python<'_>) -> String {
        let size = self.data.bind(py).as_bytes().len();
        format!(
            "FitResult(format='{}', width={}, height={}, size={}, quality_used={:.2}, reached_target={})",
            self.format, self.width, self.height, size, self.quality_used, self.reached_target
        )
    }
}

/// Compress an identity photo.
///
/// Args:
///     data: Raw image bytes (JPEG, PNG, or WebP)
///     preset: "qr-code", "qr-code-match", "print", or "display" (optional, sets all defaults)
///     max_dimension: Maximum output dimension in pixels (overrides preset, default: 48)
///     quality: Compression quality 0.0–1.0 (overrides preset, default: 0.6)
///     grayscale: Convert to grayscale (overrides preset, default: False)
///     crop_mode: "heuristic", "none", or "face-detection" (overrides preset)
///     output_format: "webp" or "jpeg" (overrides preset)
///     face_margin: Face detection crop margin (overrides preset, default: 2.0)
///
/// Returns:
///     CompressResult with attributes: data (bytes), format (str), width (int),
///                     height (int), original_size (int), face_bounds (FaceBounds or None)
///
/// Raises:
///     DecodeError: If the input cannot be decoded as an image.
///     UnsupportedFormatError: If the input format is not JPEG, PNG, or WebP.
///     EncodeError: If the output image cannot be encoded.
///     InvalidQualityError: If quality is outside 0.0–1.0.
///     InvalidMaxDimensionError: If max_dimension is 0.
///     ValueError: If preset, crop_mode, or output_format string is unrecognized.
#[pyfunction]
#[pyo3(signature = (data, *, preset=None, max_dimension=None, quality=None, grayscale=None, crop_mode=None, output_format=None, face_margin=None))]
#[allow(clippy::too_many_arguments)]
fn compress(
    py: Python<'_>,
    data: Vec<u8>,
    preset: Option<&str>,
    max_dimension: Option<u32>,
    quality: Option<f32>,
    grayscale: Option<bool>,
    crop_mode: Option<&str>,
    output_format: Option<&str>,
    face_margin: Option<f32>,
) -> PyResult<CompressResult> {
    let mut compressor = PhotoCompressor::new(data).map_err(to_py_err)?;

    if let Some(p) = preset {
        compressor = compressor.preset(string_to_preset(p)?);
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
        compressor = compressor.crop_mode(string_to_crop_mode(mode)?);
    }
    if let Some(fmt) = output_format {
        compressor = compressor.format(string_to_format(fmt)?);
    }
    if let Some(margin) = face_margin {
        compressor = compressor.face_margin(margin);
    }

    let result = compressor.compress().map_err(to_py_err)?;

    let face_bounds = result.face_bounds.as_ref().map(|fb| FaceBounds {
        x: fb.x,
        y: fb.y,
        width: fb.width,
        height: fb.height,
        confidence: fb.confidence,
    });

    Ok(CompressResult {
        data: Py::from(pyo3::types::PyBytes::new(py, &result.data)),
        format: format_to_string(&result.format).to_string(),
        width: result.width,
        height: result.height,
        original_size: result.original_size,
        face_bounds,
    })
}

/// Compress an identity photo to fit within a byte budget.
///
/// Uses binary search over quality (8 iterations) to find the highest quality
/// that produces output within `max_bytes`.
///
/// Args:
///     data: Raw image bytes (JPEG, PNG, or WebP)
///     max_bytes: Maximum output size in bytes
///     preset: "qr-code", "qr-code-match", "print", or "display" (optional, sets all defaults)
///     max_dimension: Maximum output dimension in pixels (overrides preset)
///     grayscale: Convert to grayscale (overrides preset)
///     crop_mode: "heuristic", "none", or "face-detection" (overrides preset)
///     output_format: "webp" or "jpeg" (overrides preset)
///     face_margin: Face detection crop margin (overrides preset, default: 2.0)
///
/// Returns:
///     FitResult with attributes: data (bytes), format (str), width (int),
///                     height (int), original_size (int), quality_used (float),
///                     reached_target (bool), face_bounds (FaceBounds or None)
///
/// Raises:
///     DecodeError: If the input cannot be decoded as an image.
///     UnsupportedFormatError: If the input format is not JPEG, PNG, or WebP.
///     EncodeError: If the output image cannot be encoded.
///     InvalidMaxDimensionError: If max_dimension is 0.
///     ValueError: If preset, crop_mode, or output_format string is unrecognized.
#[pyfunction]
#[pyo3(signature = (data, max_bytes, *, preset=None, max_dimension=None, grayscale=None, crop_mode=None, output_format=None, face_margin=None))]
#[allow(clippy::too_many_arguments)]
fn compress_to_fit(
    py: Python<'_>,
    data: Vec<u8>,
    max_bytes: usize,
    preset: Option<&str>,
    max_dimension: Option<u32>,
    grayscale: Option<bool>,
    crop_mode: Option<&str>,
    output_format: Option<&str>,
    face_margin: Option<f32>,
) -> PyResult<FitResult> {
    let mut compressor = PhotoCompressor::new(data).map_err(to_py_err)?;

    if let Some(p) = preset {
        compressor = compressor.preset(string_to_preset(p)?);
    }
    if let Some(dim) = max_dimension {
        compressor = compressor.max_dimension(dim);
    }
    if let Some(g) = grayscale {
        compressor = compressor.grayscale(g);
    }
    if let Some(mode) = crop_mode {
        compressor = compressor.crop_mode(string_to_crop_mode(mode)?);
    }
    if let Some(fmt) = output_format {
        compressor = compressor.format(string_to_format(fmt)?);
    }
    if let Some(margin) = face_margin {
        compressor = compressor.face_margin(margin);
    }

    let result = compressor.compress_to_fit(max_bytes).map_err(to_py_err)?;

    let face_bounds = result.photo.face_bounds.as_ref().map(|fb| FaceBounds {
        x: fb.x,
        y: fb.y,
        width: fb.width,
        height: fb.height,
        confidence: fb.confidence,
    });

    Ok(FitResult {
        data: Py::from(pyo3::types::PyBytes::new(py, &result.photo.data)),
        format: format_to_string(&result.photo.format).to_string(),
        width: result.photo.width,
        height: result.photo.height,
        original_size: result.photo.original_size,
        face_bounds,
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}

#[pymodule]
fn idphoto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("IdPhotoError", m.py().get_type::<IdPhotoException>())?;
    m.add("DecodeError", m.py().get_type::<DecodeError>())?;
    m.add(
        "UnsupportedFormatError",
        m.py().get_type::<UnsupportedFormatError>(),
    )?;
    m.add(
        "ZeroDimensionsError",
        m.py().get_type::<ZeroDimensionsError>(),
    )?;
    m.add("EncodeError", m.py().get_type::<EncodeError>())?;
    m.add(
        "InvalidQualityError",
        m.py().get_type::<InvalidQualityError>(),
    )?;
    m.add(
        "InvalidMaxDimensionError",
        m.py().get_type::<InvalidMaxDimensionError>(),
    )?;
    m.add_class::<CompressResult>()?;
    m.add_class::<FitResult>()?;
    m.add_class::<FaceBounds>()?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(compress_to_fit, m)?)?;
    Ok(())
}
