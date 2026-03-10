use std::cell::RefCell;
use std::sync::LazyLock;

use crate::face_detector::{FaceBounds, FaceDetector};

/// Parsed SeetaFace model, loaded once from the bundled binary data.
static SEETAFACE_MODEL: LazyLock<rustface::Model> = LazyLock::new(|| {
    let model_data: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../model/seeta_fd_frontal_v1.0.bin"
    ));
    rustface::read_model(std::io::Cursor::new(model_data))
        .expect("failed to load bundled SeetaFace model")
});

thread_local! {
    /// Per-thread cached detector, avoids cloning the model on every call.
    static DETECTOR: RefCell<Option<Box<dyn rustface::Detector>>> = const { RefCell::new(None) };
}

fn with_detector<R>(f: impl FnOnce(&mut dyn rustface::Detector) -> R) -> R {
    DETECTOR.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let mut det = rustface::create_detector_with_model(SEETAFACE_MODEL.clone());
            det.set_min_face_size(20);
            det.set_score_thresh(2.0);
            det.set_pyramid_scale_factor(0.8);
            det.set_slide_window_step(4, 4);
            *opt = Some(det);
        }
        f(opt.as_deref_mut().unwrap())
    })
}

/// Face detector backed by the `rustface` crate (SeetaFace engine).
///
/// The bundled SeetaFace model is parsed once (on first use) and cached
/// in a process-wide static. Construction is free after the first call.
pub struct RustfaceDetector;

impl RustfaceDetector {
    /// Create a new detector with the bundled SeetaFace model.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RustfaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetector for RustfaceDetector {
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
        let faces = with_detector(|detector| {
            detector.detect(&rustface::ImageData::new(gray, width, height))
        });

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
                    landmarks: None,
                }
            })
            .collect()
    }
}
