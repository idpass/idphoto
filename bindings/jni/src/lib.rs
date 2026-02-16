uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum IdPhotoError {
    #[error("failed to decode image: {message}")]
    DecodeError { message: String },
    #[error("unsupported image format")]
    UnsupportedFormat,
    #[error("image dimensions are zero")]
    ZeroDimensions,
    #[error("failed to encode image: {message}")]
    EncodeError { message: String },
    #[error("invalid quality value: {value}")]
    InvalidQuality { value: f32 },
    #[error("max dimension must be > 0")]
    InvalidMaxDimension,
}

impl From<idphoto::IdPhotoError> for IdPhotoError {
    fn from(e: idphoto::IdPhotoError) -> Self {
        match e {
            idphoto::IdPhotoError::DecodeError(msg) => IdPhotoError::DecodeError { message: msg },
            idphoto::IdPhotoError::UnsupportedFormat => IdPhotoError::UnsupportedFormat,
            idphoto::IdPhotoError::ZeroDimensions => IdPhotoError::ZeroDimensions,
            idphoto::IdPhotoError::EncodeError(msg) => IdPhotoError::EncodeError { message: msg },
            idphoto::IdPhotoError::InvalidQuality(q) => IdPhotoError::InvalidQuality { value: q },
            idphoto::IdPhotoError::InvalidMaxDimension => IdPhotoError::InvalidMaxDimension,
        }
    }
}

#[derive(Debug, uniffi::Enum)]
pub enum Preset {
    QrCode,
    QrCodeMatch,
    Print,
    Display,
}

impl From<Preset> for idphoto::Preset {
    fn from(preset: Preset) -> Self {
        match preset {
            Preset::QrCode => idphoto::Preset::QrCode,
            Preset::QrCodeMatch => idphoto::Preset::QrCodeMatch,
            Preset::Print => idphoto::Preset::Print,
            Preset::Display => idphoto::Preset::Display,
        }
    }
}

#[derive(Debug, uniffi::Enum)]
pub enum CropMode {
    Heuristic,
    FaceDetection,
    None,
}

impl From<CropMode> for idphoto::CropMode {
    fn from(mode: CropMode) -> Self {
        match mode {
            CropMode::Heuristic => idphoto::CropMode::Heuristic,
            CropMode::FaceDetection => idphoto::CropMode::FaceDetection,
            CropMode::None => idphoto::CropMode::None,
        }
    }
}

#[derive(Debug, uniffi::Enum)]
pub enum OutputFormat {
    Webp,
    Jpeg,
}

impl From<OutputFormat> for idphoto::OutputFormat {
    fn from(format: OutputFormat) -> Self {
        match format {
            OutputFormat::Webp => idphoto::OutputFormat::Webp,
            OutputFormat::Jpeg => idphoto::OutputFormat::Jpeg,
        }
    }
}

impl From<idphoto::OutputFormat> for OutputFormat {
    fn from(format: idphoto::OutputFormat) -> Self {
        match format {
            idphoto::OutputFormat::Webp => OutputFormat::Webp,
            idphoto::OutputFormat::Jpeg => OutputFormat::Jpeg,
        }
    }
}

#[derive(Debug, uniffi::Record)]
pub struct FaceBounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
}

#[derive(Debug, uniffi::Record)]
pub struct CompressedPhoto {
    pub data: Vec<u8>,
    pub format: OutputFormat,
    pub width: u32,
    pub height: u32,
    pub original_size: u64,
    pub face_bounds: Option<FaceBounds>,
}

#[derive(Debug, uniffi::Record)]
pub struct FitResult {
    pub photo: CompressedPhoto,
    pub quality_used: f32,
    pub reached_target: bool,
}

/// Options for configuring photo compression. All fields are optional;
/// unset fields use either the preset defaults or the library defaults.
#[derive(Debug, uniffi::Record)]
pub struct CompressOptions {
    pub preset: Option<Preset>,
    pub max_dimension: Option<u32>,
    pub quality: Option<f32>,
    pub grayscale: Option<bool>,
    pub crop_mode: Option<CropMode>,
    pub format: Option<OutputFormat>,
    pub face_margin: Option<f32>,
}

fn convert_face_bounds(bounds: &idphoto::FaceBounds) -> FaceBounds {
    FaceBounds {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
        confidence: bounds.confidence,
    }
}

fn convert_compressed_photo(result: idphoto::CompressedPhoto) -> CompressedPhoto {
    CompressedPhoto {
        data: result.data,
        format: result.format.into(),
        width: result.width,
        height: result.height,
        original_size: result.original_size as u64,
        face_bounds: result.face_bounds.as_ref().map(convert_face_bounds),
    }
}

/// Apply `CompressOptions` to a `PhotoCompressor`, returning the configured compressor.
/// Preset is applied first (if set), then individual overrides.
fn apply_options(
    compressor: idphoto::PhotoCompressor,
    options: CompressOptions,
) -> idphoto::PhotoCompressor {
    let compressor = match options.preset {
        Some(preset) => compressor.preset(preset.into()),
        None => compressor,
    };
    let compressor = match options.max_dimension {
        Some(dim) => compressor.max_dimension(dim),
        None => compressor,
    };
    let compressor = match options.quality {
        Some(q) => compressor.quality(q),
        None => compressor,
    };
    let compressor = match options.grayscale {
        Some(g) => compressor.grayscale(g),
        None => compressor,
    };
    let compressor = match options.crop_mode {
        Some(mode) => compressor.crop_mode(mode.into()),
        None => compressor,
    };
    let compressor = match options.format {
        Some(fmt) => compressor.format(fmt.into()),
        None => compressor,
    };
    let compressor = match options.face_margin {
        Some(margin) => compressor.face_margin(margin),
        None => compressor,
    };
    compressor
}

/// Returns a `CompressOptions` with all fields set to `None`,
/// suitable as a starting point for customisation.
#[uniffi::export]
pub fn default_compress_options() -> CompressOptions {
    CompressOptions {
        preset: None,
        max_dimension: None,
        quality: None,
        grayscale: None,
        crop_mode: None,
        format: None,
        face_margin: None,
    }
}

/// Compress a photo with the given options.
///
/// All option fields are optional. Set `preset` to use a pre-configured profile,
/// then override individual fields as needed. Unset fields use library defaults.
#[uniffi::export]
pub fn compress(input: Vec<u8>, options: CompressOptions) -> Result<CompressedPhoto, IdPhotoError> {
    let compressor = idphoto::PhotoCompressor::new(input)?;
    let compressor = apply_options(compressor, options);
    let result = compressor.compress()?;
    Ok(convert_compressed_photo(result))
}

/// Compress a photo to fit within a byte budget.
///
/// Uses binary search over quality to find the highest quality that fits
/// within `max_bytes`. All option fields are optional.
#[uniffi::export]
pub fn compress_to_fit(
    input: Vec<u8>,
    max_bytes: u64,
    options: CompressOptions,
) -> Result<FitResult, IdPhotoError> {
    let compressor = idphoto::PhotoCompressor::new(input)?;
    let compressor = apply_options(compressor, options);
    let result = compressor.compress_to_fit(max_bytes as usize)?;

    Ok(FitResult {
        photo: convert_compressed_photo(result.photo),
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}
