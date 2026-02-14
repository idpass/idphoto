use idphoto::{CropMode, FaceBounds, FaceDetector, OutputFormat, PhotoCompressor, Preset};

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures");

fn load_fixture(name: &str) -> Vec<u8> {
    let path = format!("{FIXTURE_DIR}/{name}");
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to load fixture {path}: {e}"))
}

#[test]
fn compress_png_to_webp() {
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input.clone())
        .unwrap()
        .max_dimension(48)
        .quality(0.6)
        .compress()
        .unwrap();

    assert!(!result.data.is_empty());
    assert!(
        result.data.len() < input.len(),
        "compressed should be smaller than original"
    );
    assert_eq!(result.original_size, input.len());
    // WebP is the default format
    assert_eq!(&result.data[0..4], b"RIFF");
    assert_eq!(&result.data[8..12], b"WEBP");
}

#[test]
fn compress_png_to_jpeg() {
    let input = load_fixture("sample_id_2.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .quality(0.8)
        .format(OutputFormat::Jpeg)
        .compress()
        .unwrap();

    assert!(!result.data.is_empty());
    // JPEG magic bytes
    assert_eq!(result.data[0], 0xFF);
    assert_eq!(result.data[1], 0xD8);
}

#[test]
fn compress_jpeg_input() {
    let input = load_fixture("sample_id_3_crop.jpg");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .compress()
        .unwrap();

    assert!(!result.data.is_empty());
}

#[test]
fn compress_webp_input() {
    let input = load_fixture("sample_id_1.webp");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .compress()
        .unwrap();

    assert!(!result.data.is_empty());
}

#[test]
fn compress_all_samples_produce_output() {
    let fixtures = [
        "sample_id_1.png",
        "sample_id_2.png",
        "sample_id_3.png",
        "sample_id_4.png",
    ];
    for name in &fixtures {
        let input = load_fixture(name);
        let result = PhotoCompressor::new(input)
            .unwrap()
            .max_dimension(48)
            .quality(0.6)
            .compress()
            .unwrap();
        assert!(!result.data.is_empty(), "failed to compress {name}");
        assert!(result.width <= 48, "{name}: width {}", result.width);
        assert!(result.height <= 64, "{name}: height {}", result.height);
    }
}

#[test]
fn compress_to_fit_500_bytes() {
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .format(OutputFormat::Jpeg)
        .compress_to_fit(500)
        .unwrap();

    if result.reached_target {
        assert!(
            result.photo.data.len() <= 500,
            "exceeded budget: {} bytes",
            result.photo.data.len()
        );
    }
    assert!(result.quality_used > 0.0);
    assert!(result.quality_used <= 1.0);
}

#[test]
fn compress_to_fit_generous_budget() {
    let input = load_fixture("sample_id_3.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .format(OutputFormat::Jpeg)
        .compress_to_fit(10_000)
        .unwrap();

    assert!(result.reached_target);
    assert!(result.photo.data.len() <= 10_000);
}

#[test]
fn grayscale_reduces_or_matches_size() {
    let input = load_fixture("sample_id_2.png");

    let color = PhotoCompressor::new(input.clone())
        .unwrap()
        .max_dimension(48)
        .format(OutputFormat::Jpeg)
        .quality(0.6)
        .compress()
        .unwrap();

    let gray = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .format(OutputFormat::Jpeg)
        .quality(0.6)
        .grayscale(true)
        .compress()
        .unwrap();

    // Grayscale should be no larger than color (usually smaller)
    assert!(
        gray.data.len() <= color.data.len() + 50, // small tolerance for encoding variance
        "grayscale ({}) significantly larger than color ({})",
        gray.data.len(),
        color.data.len()
    );
}

#[test]
fn no_crop_preserves_aspect_ratio() {
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(64)
        .crop_mode(CropMode::None)
        .compress()
        .unwrap();

    // Without crop, aspect ratio of the original is preserved
    // The larger dimension should be exactly 64
    let max_dim = result.width.max(result.height);
    assert_eq!(max_dim, 64);
}

#[test]
fn heuristic_crop_produces_portrait() {
    let input = load_fixture("sample_id_4.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(48)
        .crop_mode(CropMode::Heuristic)
        .compress()
        .unwrap();

    // 3:4 portrait: height should be greater than width
    assert!(
        result.height >= result.width,
        "expected portrait: {}x{}",
        result.width,
        result.height
    );
}

#[test]
fn higher_max_dimension_produces_larger_output() {
    let input = load_fixture("sample_id_1.png");

    let small = PhotoCompressor::new(input.clone())
        .unwrap()
        .max_dimension(32)
        .format(OutputFormat::Jpeg)
        .quality(0.8)
        .compress()
        .unwrap();

    let large = PhotoCompressor::new(input)
        .unwrap()
        .max_dimension(96)
        .format(OutputFormat::Jpeg)
        .quality(0.8)
        .compress()
        .unwrap();

    assert!(
        large.data.len() > small.data.len(),
        "96px ({} bytes) should be larger than 32px ({} bytes)",
        large.data.len(),
        small.data.len()
    );
}

#[test]
fn preset_qr_code_with_real_photo() {
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .preset(Preset::QrCode)
        .compress()
        .unwrap();

    // QR preset: 48px max, WebP, grayscale, portrait crop
    assert!(result.width <= 48);
    assert!(result.height <= 64);
    assert_eq!(&result.data[0..4], b"RIFF");
    assert_eq!(&result.data[8..12], b"WEBP");
    // Should be very small — suitable for QR embedding
    assert!(
        result.data.len() < 2000,
        "QR photo too large: {} bytes",
        result.data.len()
    );
}

#[test]
fn preset_print_with_real_photo() {
    let input = load_fixture("sample_id_2.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .preset(Preset::Print)
        .compress()
        .unwrap();

    // Print preset: 400px max, JPEG, color
    assert_eq!(result.data[0], 0xFF);
    assert_eq!(result.data[1], 0xD8);
    assert!(
        result.height >= 200,
        "print should be large: {}px",
        result.height
    );
}

#[test]
fn preset_display_with_real_photo() {
    let input = load_fixture("sample_id_3.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .preset(Preset::Display)
        .compress()
        .unwrap();

    // Display preset: 200px max, JPEG
    assert_eq!(result.data[0], 0xFF);
    assert!(result.width <= 200);
    assert!(result.height <= 200);
}

#[test]
fn preset_qr_smaller_than_print_with_real_photos() {
    let input = load_fixture("sample_id_4.png");

    let qr = PhotoCompressor::new(input.clone())
        .unwrap()
        .preset(Preset::QrCode)
        .compress()
        .unwrap();

    let print = PhotoCompressor::new(input)
        .unwrap()
        .preset(Preset::Print)
        .compress()
        .unwrap();

    assert!(
        qr.data.len() < print.data.len(),
        "QR ({} bytes) should be much smaller than Print ({} bytes)",
        qr.data.len(),
        print.data.len()
    );
}

#[test]
fn all_presets_work_on_all_samples() {
    let fixtures = [
        "sample_id_1.png",
        "sample_id_2.png",
        "sample_id_3.png",
        "sample_id_4.png",
    ];
    let presets = [
        Preset::QrCode,
        Preset::QrCodeMatch,
        Preset::Print,
        Preset::Display,
    ];

    for name in &fixtures {
        let input = load_fixture(name);
        for preset in &presets {
            let result = PhotoCompressor::new(input.clone())
                .unwrap()
                .preset(preset.clone())
                .compress();
            assert!(
                result.is_ok(),
                "preset {:?} failed on {}: {}",
                preset,
                name,
                result.unwrap_err()
            );
        }
    }
}

/// Mock face detector for integration tests.
struct MockDetector {
    faces: Vec<FaceBounds>,
}

impl MockDetector {
    fn with_face(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            faces: vec![FaceBounds {
                x,
                y,
                width,
                height,
                confidence: 10.0,
            }],
        }
    }
}

impl FaceDetector for MockDetector {
    fn detect(&self, _gray: &[u8], _width: u32, _height: u32) -> Vec<FaceBounds> {
        self.faces.clone()
    }
}

#[test]
fn custom_face_detector_via_builder() {
    let input = load_fixture("sample_id_1.png");
    let detector = MockDetector::with_face(100.0, 150.0, 80.0, 80.0);
    let result = PhotoCompressor::new(input)
        .unwrap()
        .crop_mode(CropMode::FaceDetection)
        .face_detector(Box::new(detector))
        .max_dimension(48)
        .compress()
        .unwrap();

    assert!(!result.data.is_empty());
    assert!(
        result.face_bounds.is_some(),
        "face_bounds should be populated with custom detector"
    );
}

#[test]
fn face_detection_mode_without_detector_produces_output() {
    let input = load_fixture("sample_id_2.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .crop_mode(CropMode::FaceDetection)
        .max_dimension(48)
        .compress()
        .unwrap();

    // Should succeed regardless of whether rustface feature is compiled in.
    // Without a detector, falls back to heuristic.
    assert!(!result.data.is_empty());
}

#[cfg(feature = "rustface")]
#[test]
fn builtin_rustface_backend_detects_faces() {
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .preset(Preset::QrCode)
        .compress()
        .unwrap();

    // With the rustface feature, the QrCode preset uses FaceDetection mode
    // and the sample photos should have detectable faces.
    assert!(!result.data.is_empty());
    assert!(
        result.face_bounds.is_some(),
        "rustface should detect a face in sample_id_1.png"
    );
}

#[cfg(feature = "rustface")]
#[test]
fn rustface_detector_can_be_used_directly() {
    let detector = idphoto::RustfaceDetector::new();
    let input = load_fixture("sample_id_1.png");
    let result = PhotoCompressor::new(input)
        .unwrap()
        .face_detector(Box::new(detector))
        .crop_mode(CropMode::FaceDetection)
        .max_dimension(48)
        .compress()
        .unwrap();

    assert!(
        result.face_bounds.is_some(),
        "explicit RustfaceDetector should detect a face"
    );
}
