use idphoto_core::{CropMode, IdPhotoError, OutputFormat, PhotoCompressor, Preset};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py_err(e: IdPhotoError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn format_to_string(format: &OutputFormat) -> &'static str {
    match format {
        OutputFormat::Webp => "webp",
        OutputFormat::Jpeg => "jpeg",
    }
}

fn string_to_preset(preset: &str) -> PyResult<Preset> {
    match preset {
        "qr" | "qr-code" => Ok(Preset::QrCode),
        "qr-match" | "qr-code-match" => Ok(Preset::QrCodeMatch),
        "print" => Ok(Preset::Print),
        "display" => Ok(Preset::Display),
        _ => Err(PyValueError::new_err(format!("unknown preset: {preset}"))),
    }
}

fn string_to_crop_mode(mode: &str) -> PyResult<CropMode> {
    match mode {
        "heuristic" => Ok(CropMode::Heuristic),
        "none" => Ok(CropMode::None),
        #[cfg(feature = "face-detection")]
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

/// Compress an identity photo.
///
/// Args:
///     input: Raw image bytes (JPEG, PNG, or WebP)
///     preset: "qr-code", "print", or "display" (optional, sets all defaults)
///     max_dimension: Maximum output dimension in pixels (overrides preset, default: 48)
///     quality: Compression quality 0.0–1.0 (overrides preset, default: 0.6)
///     grayscale: Convert to grayscale (overrides preset, default: False)
///     crop_mode: "heuristic", "none", or "face-detection" (overrides preset)
///     format: "webp" or "jpeg" (overrides preset)
///
/// Returns:
///     dict with keys: data (bytes), format (str), width (int), height (int), original_size (int)
#[pyfunction]
#[pyo3(signature = (input, *, preset=None, max_dimension=None, quality=None, grayscale=None, crop_mode=None, format=None))]
#[allow(clippy::too_many_arguments)]
fn compress(
    py: Python<'_>,
    input: Vec<u8>,
    preset: Option<&str>,
    max_dimension: Option<u32>,
    quality: Option<f32>,
    grayscale: Option<bool>,
    crop_mode: Option<&str>,
    format: Option<&str>,
) -> PyResult<Py<PyDict>> {
    let mut compressor = PhotoCompressor::new(input).map_err(to_py_err)?;

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
    if let Some(fmt) = format {
        compressor = compressor.format(string_to_format(fmt)?);
    }

    let result = compressor.compress().map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("data", pyo3::types::PyBytes::new(py, &result.data))?;
    dict.set_item("format", format_to_string(&result.format))?;
    dict.set_item("width", result.width)?;
    dict.set_item("height", result.height)?;
    dict.set_item("original_size", result.original_size)?;
    Ok(dict.into())
}

/// Compress an identity photo to fit within a byte budget.
///
/// Uses binary search over quality (8 iterations) to find the highest quality
/// that produces output within `max_bytes`.
///
/// Args:
///     input: Raw image bytes (JPEG, PNG, or WebP)
///     max_bytes: Maximum output size in bytes
///     preset: "qr-code", "print", or "display" (optional, sets all defaults)
///     max_dimension: Maximum output dimension in pixels (overrides preset)
///     grayscale: Convert to grayscale (overrides preset)
///     crop_mode: "heuristic", "none", or "face-detection" (overrides preset)
///     format: "webp" or "jpeg" (overrides preset)
///
/// Returns:
///     dict with keys: data (bytes), format (str), width (int), height (int),
///                     original_size (int), quality_used (float), reached_target (bool)
#[pyfunction]
#[pyo3(signature = (input, max_bytes, *, preset=None, max_dimension=None, grayscale=None, crop_mode=None, format=None))]
#[allow(clippy::too_many_arguments)]
fn compress_to_fit(
    py: Python<'_>,
    input: Vec<u8>,
    max_bytes: usize,
    preset: Option<&str>,
    max_dimension: Option<u32>,
    grayscale: Option<bool>,
    crop_mode: Option<&str>,
    format: Option<&str>,
) -> PyResult<Py<PyDict>> {
    let mut compressor = PhotoCompressor::new(input).map_err(to_py_err)?;

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
    if let Some(fmt) = format {
        compressor = compressor.format(string_to_format(fmt)?);
    }

    let result = compressor.compress_to_fit(max_bytes).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("data", pyo3::types::PyBytes::new(py, &result.photo.data))?;
    dict.set_item("format", format_to_string(&result.photo.format))?;
    dict.set_item("width", result.photo.width)?;
    dict.set_item("height", result.photo.height)?;
    dict.set_item("original_size", result.photo.original_size)?;
    dict.set_item("quality_used", result.quality_used)?;
    dict.set_item("reached_target", result.reached_target)?;
    Ok(dict.into())
}

#[pymodule]
fn idphoto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(compress_to_fit, m)?)?;
    Ok(())
}
