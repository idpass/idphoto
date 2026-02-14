use thiserror::Error;

/// Errors that can occur while decoding, processing, or encoding photos.
#[derive(Debug, Error)]
pub enum IdPhotoError {
    /// The input bytes could not be decoded as a supported image.
    #[error("failed to decode image: {0}")]
    DecodeError(String),

    /// The input format is not one of JPEG, PNG, or WebP.
    #[error("unsupported image format")]
    UnsupportedFormat,

    /// Decoded image has zero width or height.
    #[error("image dimensions are zero")]
    ZeroDimensions,

    /// The output image could not be encoded.
    #[error("failed to encode image: {0}")]
    EncodeError(String),

    /// Compression quality is outside the valid 0.0..=1.0 range.
    #[error("quality must be between 0.0 and 1.0, got {0}")]
    InvalidQuality(f32),

    /// Maximum output dimension must be greater than zero.
    #[error("max dimension must be > 0")]
    InvalidMaxDimension,
}
