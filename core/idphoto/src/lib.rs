//! Identity photo processing: crop, resize, and compress photos for ID documents and QR codes.
//!
//! # Example
//!
//! ```no_run
//! use idphoto::PhotoCompressor;
//!
//! let raw_bytes = std::fs::read("photo.jpg").unwrap();
//! let result = PhotoCompressor::new(raw_bytes)
//!     .unwrap()
//!     .max_dimension(48)
//!     .quality(0.6)
//!     .compress()
//!     .unwrap();
//! println!("Compressed: {} bytes", result.data.len());
//! ```
#![warn(missing_docs)]

mod compress;
mod crop;
mod error;
/// Face detection traits and data types.
pub mod face_detector;
#[cfg(feature = "rustface")]
/// Built-in SeetaFace-based face detector backend.
pub mod rustface_backend;
mod webp_strip;

/// Error type returned by idphoto operations.
pub use error::IdPhotoError;
/// Face detection trait and face bounding-box type.
pub use face_detector::{FaceBounds, FaceDetector};
#[cfg(feature = "rustface")]
/// Built-in detector that loads the bundled SeetaFace model.
pub use rustface_backend::RustfaceDetector;

/// How to crop the input image before resizing.
#[derive(Debug, Clone, Default)]
pub enum CropMode {
    /// 3:4 portrait, upper-center bias — good for passport-style photos.
    #[default]
    Heuristic,

    /// Use face detection to center on the face, with fallback to Heuristic.
    FaceDetection,

    /// No crop — resize maintaining the original aspect ratio.
    None,
}

/// Output image format.
#[derive(Debug, Clone, Default)]
pub enum OutputFormat {
    /// WebP lossy encoding (generally smaller at equivalent quality).
    #[default]
    Webp,

    /// JPEG encoding.
    Jpeg,
}

/// Result of a single compression operation.
#[derive(Debug, Clone)]
pub struct CompressedPhoto {
    /// The compressed image bytes.
    pub data: Vec<u8>,

    /// The output format used.
    pub format: OutputFormat,

    /// Width of the output image in pixels.
    pub width: u32,

    /// Height of the output image in pixels.
    pub height: u32,

    /// Size of the original input in bytes.
    pub original_size: usize,

    /// Bounding box of the detected face in output image coordinates, if any.
    pub face_bounds: Option<FaceBounds>,
}

/// Result of a `compress_to_fit` operation with byte budget targeting.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The compressed photo.
    pub photo: CompressedPhoto,

    /// The quality level used (0.0–1.0).
    pub quality_used: f32,

    /// Whether the target byte budget was achieved.
    /// `false` if even minimum quality exceeds `max_bytes`.
    pub reached_target: bool,
}

/// Pre-configured settings for common identity photo use cases.
///
/// Apply a preset with [`PhotoCompressor::preset`], then override individual
/// settings as needed. Presets set all parameters — any `.preset()` call
/// replaces the full configuration.
#[derive(Debug, Clone)]
pub enum Preset {
    /// For manual visual verification in QR codes.
    ///
    /// - 48px max dimension (3:4 portrait → 36×48)
    /// - WebP lossy at quality 0.6 with ICC stripping
    /// - Grayscale (single channel reduces payload)
    /// - Face detection crop, ID-photo framing (face + hair + shoulders)
    QrCode,

    /// For algorithmic face matching in QR codes.
    ///
    /// - 48px max dimension (3:4 portrait → 36×48)
    /// - WebP lossy at quality 0.6 with ICC stripping
    /// - Grayscale (single channel reduces payload)
    /// - Face detection crop, tight framing (face only, minimal margin)
    QrCodeMatch,

    /// High-quality output for printed ID cards and verification documents.
    ///
    /// - 400px max dimension (3:4 portrait → 300×400)
    /// - JPEG at 0.9 quality
    /// - Full color
    /// - Face detection crop, ID-photo framing
    Print,

    /// Mid-resolution for on-screen display and digital verification.
    ///
    /// - 200px max dimension (3:4 portrait → 150×200)
    /// - JPEG at 0.75 quality
    /// - Full color
    /// - Face detection crop, ID-photo framing
    Display,
}

/// Default face margin for ID-photo framing (face + hair + shoulders).
const FACE_MARGIN_ID_PHOTO: f32 = 2.0;

/// Tight face margin for algorithmic face matching (face only).
const FACE_MARGIN_MATCH: f32 = 1.3;

/// Builder for compressing identity photos.
///
/// Decodes the input image on construction, then applies crop, resize,
/// and compression with configurable parameters.
pub struct PhotoCompressor {
    input: Vec<u8>,
    max_dimension: u32,
    quality: f32,
    grayscale: bool,
    crop_mode: CropMode,
    format: OutputFormat,
    /// Multiplier for face-detection crop: crop_height = face_height × face_margin.
    /// Higher values include more context (hair, shoulders), lower values zoom tighter.
    face_margin: f32,
    /// User-provided face detector. When `None`, the built-in rustface backend is
    /// used (if compiled with the `rustface` feature), or detection is skipped.
    detector: Option<Box<dyn FaceDetector>>,
}

impl PhotoCompressor {
    /// The default crop mode: face detection with automatic fallback to heuristic.
    fn default_crop_mode() -> CropMode {
        CropMode::FaceDetection
    }

    /// Create a new compressor from raw image bytes (JPEG, PNG, or WebP).
    pub fn new(input: Vec<u8>) -> Result<Self, IdPhotoError> {
        // Validate that the input can be decoded
        compress::detect_format(&input)?;

        Ok(Self {
            input,
            max_dimension: 48,
            quality: 0.6,
            grayscale: false,
            crop_mode: Self::default_crop_mode(),
            format: OutputFormat::default(),
            face_margin: FACE_MARGIN_ID_PHOTO,
            detector: None,
        })
    }

    /// Apply a preset configuration. Individual settings can be overridden
    /// after this call.
    ///
    /// ```no_run
    /// use idphoto::{PhotoCompressor, Preset};
    ///
    /// let bytes = std::fs::read("photo.jpg").unwrap();
    ///
    /// // QR-optimized: tiny WebP, grayscale, ICC-stripped
    /// let qr = PhotoCompressor::new(bytes.clone()).unwrap()
    ///     .preset(Preset::QrCode)
    ///     .compress().unwrap();
    ///
    /// // Print-quality: large JPEG, full color
    /// let print = PhotoCompressor::new(bytes).unwrap()
    ///     .preset(Preset::Print)
    ///     .compress().unwrap();
    /// ```
    pub fn preset(mut self, preset: Preset) -> Self {
        match preset {
            Preset::QrCode => {
                self.max_dimension = 48;
                self.quality = 0.6;
                self.grayscale = true;
                self.crop_mode = Self::default_crop_mode();
                self.face_margin = FACE_MARGIN_ID_PHOTO;
                self.format = OutputFormat::Webp;
            }
            Preset::QrCodeMatch => {
                self.max_dimension = 48;
                self.quality = 0.6;
                self.grayscale = true;
                self.crop_mode = Self::default_crop_mode();
                self.face_margin = FACE_MARGIN_MATCH;
                self.format = OutputFormat::Webp;
            }
            Preset::Print => {
                self.max_dimension = 400;
                self.quality = 0.9;
                self.grayscale = false;
                self.crop_mode = Self::default_crop_mode();
                self.face_margin = FACE_MARGIN_ID_PHOTO;
                self.format = OutputFormat::Jpeg;
            }
            Preset::Display => {
                self.max_dimension = 200;
                self.quality = 0.75;
                self.grayscale = false;
                self.crop_mode = Self::default_crop_mode();
                self.face_margin = FACE_MARGIN_ID_PHOTO;
                self.format = OutputFormat::Jpeg;
            }
        }
        self
    }

    /// Set the maximum output dimension in pixels (default: 48).
    ///
    /// For portrait crop mode, this becomes the width and the height is
    /// calculated from the 3:4 aspect ratio. For no-crop mode, the larger
    /// dimension is constrained to this value.
    pub fn max_dimension(mut self, dimension: u32) -> Self {
        self.max_dimension = dimension;
        self
    }

    /// Set the compression quality from 0.0 (lowest) to 1.0 (highest).
    /// Default: 0.6.
    pub fn quality(mut self, quality: f32) -> Self {
        self.quality = quality;
        self
    }

    /// Enable or disable grayscale conversion (default: false).
    pub fn grayscale(mut self, enable: bool) -> Self {
        self.grayscale = enable;
        self
    }

    /// Set the crop mode (default: `CropMode::FaceDetection`).
    pub fn crop_mode(mut self, mode: CropMode) -> Self {
        self.crop_mode = mode;
        self
    }

    /// Set the output format (default: `OutputFormat::Webp`).
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the face detection crop margin (default: 2.0).
    ///
    /// Controls how tightly the face-detection crop frames the face.
    /// The crop height is `face_height × margin`. Only applies when
    /// `CropMode::FaceDetection` is active.
    ///
    /// - `2.0` — ID-photo framing: face + hair + shoulders
    /// - `1.3` — tight face crop for algorithmic matching
    pub fn face_margin(mut self, margin: f32) -> Self {
        self.face_margin = margin;
        self
    }

    /// Provide a custom face detector implementation.
    ///
    /// When set, this detector is used instead of the built-in rustface backend.
    /// This allows integrating ONNX, dlib, or any other face detection engine.
    ///
    /// ```no_run
    /// use idphoto::{PhotoCompressor, FaceDetector, FaceBounds};
    ///
    /// struct MyDetector;
    /// impl FaceDetector for MyDetector {
    ///     fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
    ///         // Your detection logic here
    ///         vec![]
    ///     }
    /// }
    ///
    /// let bytes = std::fs::read("photo.jpg").unwrap();
    /// let result = PhotoCompressor::new(bytes).unwrap()
    ///     .face_detector(Box::new(MyDetector))
    ///     .compress().unwrap();
    /// ```
    pub fn face_detector(mut self, detector: Box<dyn FaceDetector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Compress the photo with the configured settings.
    pub fn compress(self) -> Result<CompressedPhoto, IdPhotoError> {
        if self.max_dimension == 0 {
            return Err(IdPhotoError::InvalidMaxDimension);
        }
        if self.quality < 0.0 || self.quality > 1.0 {
            return Err(IdPhotoError::InvalidQuality(self.quality));
        }

        compress::compress_pipeline(
            &self.input,
            self.max_dimension,
            self.quality,
            self.grayscale,
            &self.crop_mode,
            &self.format,
            self.face_margin,
            self.detector.as_deref(),
        )
    }

    /// Binary-search over quality to compress within a byte budget.
    ///
    /// Runs up to 8 iterations of binary search (covers 10%-100% at <1% precision).
    /// Returns the highest quality that fits within `max_bytes`, or the minimum-quality
    /// result with `reached_target: false` if the target is unreachable.
    pub fn compress_to_fit(self, max_bytes: usize) -> Result<FitResult, IdPhotoError> {
        if self.max_dimension == 0 {
            return Err(IdPhotoError::InvalidMaxDimension);
        }

        let mut low: f32 = 0.1;
        let mut high: f32 = 1.0;
        let mut best_result: Option<CompressedPhoto> = None;
        let mut best_quality = low;

        for _ in 0..8 {
            let mid = (low + high) / 2.0;

            let result = compress::compress_pipeline(
                &self.input,
                self.max_dimension,
                mid,
                self.grayscale,
                &self.crop_mode,
                &self.format,
                self.face_margin,
                self.detector.as_deref(),
            )?;

            if result.data.len() <= max_bytes {
                best_quality = mid;
                best_result = Some(result);
                low = mid;
            } else {
                high = mid;
            }
        }

        if let Some(photo) = best_result {
            return Ok(FitResult {
                photo,
                quality_used: best_quality,
                reached_target: true,
            });
        }

        // Even minimum quality exceeds target — return best-effort result
        let fallback = compress::compress_pipeline(
            &self.input,
            self.max_dimension,
            0.1,
            self.grayscale,
            &self.crop_mode,
            &self.format,
            self.face_margin,
            self.detector.as_deref(),
        )?;

        Ok(FitResult {
            photo: fallback,
            quality_used: 0.1,
            reached_target: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_png(width: u32, height: u32) -> Vec<u8> {
        use image::codecs::png::PngEncoder;
        use image::ImageEncoder;
        use image::RgbImage;

        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = image::Rgb([
                (x * 255 / width.max(1)) as u8,
                (y * 255 / height.max(1)) as u8,
                128,
            ]);
        }
        let mut buffer = Vec::new();
        let encoder = PngEncoder::new(&mut buffer);
        encoder
            .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgb8)
            .unwrap();
        buffer
    }

    #[test]
    fn builder_defaults() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png).unwrap().compress().unwrap();
        assert!(!result.data.is_empty());
        // Default max_dimension=48 with 3:4 crop → 36x48 or 48x64
        // With heuristic crop on 200x300 (aspect 0.67 < 0.75):
        // crop: width=200, height=267 → resize: max_dim=48 constrains height to 48, width=36
        assert!(result.width <= 48);
        assert!(result.height <= 64);
    }

    #[test]
    fn builder_with_jpeg_format() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .format(OutputFormat::Jpeg)
            .compress()
            .unwrap();
        assert_eq!(result.data[0], 0xFF);
        assert_eq!(result.data[1], 0xD8);
    }

    #[test]
    fn builder_with_grayscale() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .grayscale(true)
            .compress()
            .unwrap();
        assert!(!result.data.is_empty());
    }

    #[test]
    fn builder_with_no_crop() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .crop_mode(CropMode::None)
            .max_dimension(64)
            .compress()
            .unwrap();
        // 200x300 → constrain to 64: height=64, width=43
        assert_eq!(result.height, 64);
        assert_eq!(result.width, 43);
    }

    #[test]
    fn builder_invalid_quality_high() {
        let png = make_test_png(100, 100);
        let result = PhotoCompressor::new(png).unwrap().quality(1.5).compress();
        assert!(result.is_err());
    }

    #[test]
    fn builder_invalid_quality_low() {
        let png = make_test_png(100, 100);
        let result = PhotoCompressor::new(png).unwrap().quality(-0.1).compress();
        assert!(result.is_err());
    }

    #[test]
    fn builder_zero_max_dimension() {
        let png = make_test_png(100, 100);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .max_dimension(0)
            .compress();
        assert!(result.is_err());
    }

    #[test]
    fn builder_invalid_input() {
        let result = PhotoCompressor::new(b"not an image".to_vec());
        assert!(result.is_err());
    }

    #[test]
    fn compress_to_fit_reaches_target() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .max_dimension(48)
            .compress_to_fit(10_000) // generous budget
            .unwrap();
        assert!(result.reached_target);
        assert!(result.photo.data.len() <= 10_000);
        assert!(result.quality_used > 0.0);
    }

    #[test]
    fn compress_to_fit_tight_budget() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .max_dimension(48)
            .compress_to_fit(500)
            .unwrap();
        // May or may not reach target depending on image content
        if result.reached_target {
            assert!(result.photo.data.len() <= 500);
        }
    }

    #[test]
    fn compress_to_fit_impossible_budget() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .max_dimension(48)
            .compress_to_fit(1) // impossibly small
            .unwrap();
        assert!(!result.reached_target);
        assert!((result.quality_used - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn original_size_is_preserved() {
        let png = make_test_png(200, 300);
        let original_len = png.len();
        let result = PhotoCompressor::new(png).unwrap().compress().unwrap();
        assert_eq!(result.original_size, original_len);
    }

    #[test]
    fn compress_produces_smaller_output() {
        let png = make_test_png(200, 300);
        let original_len = png.len();
        let result = PhotoCompressor::new(png).unwrap().compress().unwrap();
        assert!(result.data.len() < original_len);
    }

    #[test]
    fn preset_qr_code() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::QrCode)
            .compress()
            .unwrap();
        // QR preset: 48px, WebP, grayscale
        assert!(result.width <= 48);
        assert!(result.height <= 64);
        assert_eq!(&result.data[0..4], b"RIFF"); // WebP
    }

    #[test]
    fn preset_print() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::Print)
            .compress()
            .unwrap();
        // Print preset: 400px, JPEG
        assert_eq!(result.data[0], 0xFF); // JPEG
        assert_eq!(result.data[1], 0xD8);
        // Source is only 200x300, so output will be smaller than 400px
        assert!(result.width <= 400);
        assert!(result.height <= 400);
    }

    #[test]
    fn preset_display() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::Display)
            .compress()
            .unwrap();
        // Display preset: 200px, JPEG
        assert_eq!(result.data[0], 0xFF); // JPEG
        assert_eq!(result.data[1], 0xD8);
        assert!(result.width <= 200);
        assert!(result.height <= 200);
    }

    #[test]
    fn preset_qr_code_match() {
        let png = make_test_png(200, 300);
        let result = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::QrCodeMatch)
            .compress()
            .unwrap();
        // QrCodeMatch preset: 48px, WebP, grayscale (same format as QrCode)
        assert!(result.width <= 48);
        assert!(result.height <= 64);
        assert_eq!(&result.data[0..4], b"RIFF"); // WebP
    }

    #[test]
    fn preset_can_be_overridden() {
        let png = make_test_png(200, 300);
        // Start with QR preset (WebP, grayscale, 48px) then override format
        let result = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::QrCode)
            .format(OutputFormat::Jpeg)
            .compress()
            .unwrap();
        // Should use JPEG despite QR preset
        assert_eq!(result.data[0], 0xFF);
        assert_eq!(result.data[1], 0xD8);
        // But keep other QR settings (48px)
        assert!(result.width <= 48);
    }

    #[test]
    fn preset_qr_produces_smallest() {
        let png = make_test_png(200, 300);

        let qr = PhotoCompressor::new(png.clone())
            .unwrap()
            .preset(Preset::QrCode)
            .format(OutputFormat::Jpeg) // use JPEG for fair comparison
            .compress()
            .unwrap();

        let print = PhotoCompressor::new(png)
            .unwrap()
            .preset(Preset::Print)
            .compress()
            .unwrap();

        assert!(
            qr.data.len() < print.data.len(),
            "QR ({} bytes) should be smaller than Print ({} bytes)",
            qr.data.len(),
            print.data.len()
        );
    }
}
