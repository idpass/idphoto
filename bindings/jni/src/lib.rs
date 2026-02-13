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
    None,
}

impl From<CropMode> for idphoto::CropMode {
    fn from(mode: CropMode) -> Self {
        match mode {
            CropMode::Heuristic => idphoto::CropMode::Heuristic,
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
pub struct CompressedPhoto {
    pub data: Vec<u8>,
    pub format: OutputFormat,
    pub width: u32,
    pub height: u32,
    pub original_size: u64,
}

#[derive(uniffi::Record)]
pub struct FitResult {
    pub photo: CompressedPhoto,
    pub quality_used: f32,
    pub reached_target: bool,
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
) -> Result<CompressedPhoto, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .max_dimension(max_dimension)
        .quality(quality)
        .grayscale(grayscale)
        .crop_mode(crop_mode.into())
        .format(format.into())
        .compress()?;

    Ok(CompressedPhoto {
        data: result.data,
        format: result.format.into(),
        width: result.width,
        height: result.height,
        original_size: result.original_size as u64,
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
) -> Result<FitResult, IdPhotoError> {
    let result = idphoto::PhotoCompressor::new(input)?
        .max_dimension(max_dimension)
        .grayscale(grayscale)
        .crop_mode(crop_mode.into())
        .format(format.into())
        .compress_to_fit(max_bytes as usize)?;

    Ok(FitResult {
        photo: CompressedPhoto {
            data: result.photo.data,
            format: result.photo.format.into(),
            width: result.photo.width,
            height: result.photo.height,
            original_size: result.photo.original_size as u64,
        },
        quality_used: result.quality_used,
        reached_target: result.reached_target,
    })
}
