use crate::face_detector::{FaceBounds, FaceDetector};

/// Face detector backed by the `rustface` crate (SeetaFace engine).
///
/// Loads the bundled SeetaFace model on construction. The model is embedded
/// in the binary via `include_bytes!`, so no external files are needed at runtime.
pub struct RustfaceDetector {
    model: rustface::Model,
}

impl RustfaceDetector {
    /// Create a new detector with the bundled SeetaFace model.
    pub fn new() -> Self {
        let model_data: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../model/seeta_fd_frontal_v1.0.bin"
        ));
        let model = rustface::read_model(std::io::Cursor::new(model_data))
            .expect("failed to load bundled SeetaFace model");
        Self { model }
    }
}

impl Default for RustfaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetector for RustfaceDetector {
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
        let mut detector = rustface::create_detector_with_model(self.model.clone());
        detector.set_min_face_size(20);
        detector.set_score_thresh(2.0);
        detector.set_pyramid_scale_factor(0.8);
        detector.set_slide_window_step(4, 4);

        let faces = detector.detect(&rustface::ImageData::new(gray, width, height));

        faces
            .iter()
            .map(|face| {
                let bbox = face.bbox();
                FaceBounds {
                    x: bbox.x() as f64,
                    y: bbox.y() as f64,
                    width: bbox.width() as f64,
                    height: bbox.height() as f64,
                    confidence: face.score(),
                }
            })
            .collect()
    }
}
