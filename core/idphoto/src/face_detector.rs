/// 5-point facial landmarks for ArcFace-style alignment.
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    /// Right eye center (x, y).
    pub right_eye: (f64, f64),
    /// Left eye center (x, y).
    pub left_eye: (f64, f64),
    /// Nose tip (x, y).
    pub nose: (f64, f64),
    /// Right mouth corner (x, y).
    pub right_mouth: (f64, f64),
    /// Left mouth corner (x, y).
    pub left_mouth: (f64, f64),
}

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
    /// 5-point facial landmarks, if provided by the detector.
    pub landmarks: Option<FaceLandmarks>,
}

/// Pluggable face detection backend.
///
/// Implement this trait to provide a custom face detector (ONNX, dlib, etc.)
/// and pass it to [`crate::PhotoCompressor::face_detector`].
pub trait FaceDetector: Send + Sync {
    /// Detect faces in a row-major grayscale buffer of `width` × `height` bytes.
    ///
    /// The buffer contains one byte per pixel in row-major order. Detectors
    /// that require color input (e.g. BGR) must convert internally.
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds>;
}
