use thiserror::Error;

#[derive(Debug, Error)]
pub enum IdPhotoError {
    #[error("failed to decode image: {0}")]
    DecodeError(String),

    #[error("unsupported image format")]
    UnsupportedFormat,

    #[error("image dimensions are zero")]
    ZeroDimensions,

    #[error("failed to encode image: {0}")]
    EncodeError(String),

    #[error("quality must be between 0.0 and 1.0, got {0}")]
    InvalidQuality(f32),

    #[error("max dimension must be > 0")]
    InvalidMaxDimension,
}
