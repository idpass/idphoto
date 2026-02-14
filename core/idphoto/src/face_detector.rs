/// Bounding box of a detected face within an image.
#[derive(Debug, Clone)]
pub struct FaceBounds {
    /// X coordinate of the top-left corner (pixels).
    pub x: f64,
    /// Y coordinate of the top-left corner (pixels).
    pub y: f64,
    /// Width of the bounding box (pixels).
    pub width: f64,
    /// Height of the bounding box (pixels).
    pub height: f64,
    /// Detection confidence score.
    pub confidence: f64,
}

/// Pluggable face detection backend.
///
/// Implement this trait to provide a custom face detector (ONNX, dlib, etc.)
/// and pass it to [`crate::PhotoCompressor::face_detector`].
pub trait FaceDetector: Send + Sync {
    /// Detect faces in a row-major grayscale buffer of `width` × `height` bytes.
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds>;
}
