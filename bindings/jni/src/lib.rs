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
    #[error("invalid quality value")]
    InvalidQuality,
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
            idphoto::IdPhotoError::InvalidQuality(_) => IdPhotoError::InvalidQuality,
            idphoto::IdPhotoError::InvalidMaxDimension => IdPhotoError::InvalidMaxDimension,
        }
    }
}

#[derive(uniffi::Enum)]
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

#[derive(uniffi::Enum)]
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

#[derive(uniffi::Enum)]
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

#[derive(uniffi::Record)]
pub struct FaceBounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
}

#[derive(uniffi::Record)]
pub struct CompressedPhoto {
    pub data: Vec<u8>,
    pub format: OutputFormat,
    pub width: u32,
    pub height: u32,
    pub original_size: u64,
    pub face_bounds: Option<FaceBounds>,
}

#[derive(uniffi::Record)]
pub struct FitResult {
    pub photo: CompressedPhoto,
    pub quality_used: f32,
    pub reached_target: bool,
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

/// Compress with a preset configuration.
#[uniffi::export]
pub fn compress_with_preset(
    input: Vec<u8>,
    preset: Preset,
) -> Result<CompressedPhoto, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .preset(preset.into())
        .compress()?;

    Ok(CompressedPhoto {
        data: result.data,
        format: result.format.into(),
        width: result.width,
        height: result.height,
        original_size: result.original_size as u64,
        face_bounds: result.face_bounds.as_ref().map(convert_face_bounds),
    })
}

/// Compress with full control over all parameters.
#[uniffi::export]
pub fn compress(
    input: Vec<u8>,
    max_dimension: u32,
    quality: f32,
    grayscale: bool,
    crop_mode: CropMode,
    format: OutputFormat,
    face_margin: f32,
) -> Result<CompressedPhoto, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .max_dimension(max_dimension)
        .quality(quality)
        .grayscale(grayscale)
        .crop_mode(crop_mode.into())
        .format(format.into())
        .face_margin(face_margin)
        .compress()?;

    Ok(CompressedPhoto {
        data: result.data,
        format: result.format.into(),
        width: result.width,
        height: result.height,
        original_size: result.original_size as u64,
        face_bounds: result.face_bounds.as_ref().map(convert_face_bounds),
    })
}

/// Compress to fit within a byte budget using a preset.
#[uniffi::export]
pub fn compress_to_fit_with_preset(
    input: Vec<u8>,
    max_bytes: u64,
    preset: Preset,
) -> Result<FitResult, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .preset(preset.into())
        .compress_to_fit(max_bytes as usize)?;

    Ok(FitResult {
        photo: CompressedPhoto {
            data: result.photo.data,
            format: result.photo.format.into(),
            width: result.photo.width,
            height: result.photo.height,
            original_size: result.photo.original_size as u64,
            face_bounds: result.photo.face_bounds.as_ref().map(convert_face_bounds),
        },
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}

/// Compress to fit within a byte budget with full control over parameters.
#[uniffi::export]
pub fn compress_to_fit(
    input: Vec<u8>,
    max_bytes: u64,
    max_dimension: u32,
    grayscale: bool,
    crop_mode: CropMode,
    format: OutputFormat,
    face_margin: f32,
) -> Result<FitResult, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .max_dimension(max_dimension)
        .grayscale(grayscale)
        .crop_mode(crop_mode.into())
        .format(format.into())
        .face_margin(face_margin)
        .compress_to_fit(max_bytes as usize)?;

    Ok(FitResult {
        photo: CompressedPhoto {
            data: result.photo.data,
            format: result.photo.format.into(),
            width: result.photo.width,
            height: result.photo.height,
            original_size: result.photo.original_size as u64,
            face_bounds: result.photo.face_bounds.as_ref().map(convert_face_bounds),
        },
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}
