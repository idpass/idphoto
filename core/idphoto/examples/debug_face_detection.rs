//! Debug face detection on sample images to see if faces are found.
//!
//! Usage:
//!   cargo run --example debug_face_detection --features face-detection

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures");

fn main() {
    let samples = [
        "sample_id_1.png",
        "sample_id_2.png",
        "sample_id_3.png",
        "sample_id_4.png",
    ];

    let model_data: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../model/seeta_fd_frontal_v1.0.bin"
    ));
    let model =
        rustface::read_model(std::io::Cursor::new(model_data)).expect("failed to load model");

    for sample in &samples {
        let input_path = format!("{FIXTURE_DIR}/{sample}");
        let input = std::fs::read(&input_path).unwrap();
        let image = image::load_from_memory(&input).unwrap();
        let gray = image::imageops::grayscale(&image);
        let (width, height) = (gray.width(), gray.height());

        println!("=== {sample} ({width}x{height}) ===");

        let mut detector = rustface::create_detector_with_model(model.clone());
        detector.set_min_face_size(20);
        detector.set_score_thresh(2.0);
        detector.set_pyramid_scale_factor(0.8);
        detector.set_slide_window_step(4, 4);

        let faces = detector.detect(&rustface::ImageData::new(gray.as_raw(), width, height));

        if faces.is_empty() {
            println!("  NO FACES DETECTED — falling back to heuristic");
        } else {
            println!("  Found {} face(s):", faces.len());
            for (i, face) in faces.iter().enumerate() {
                let bbox = face.bbox();
                println!(
                    "    face {i}: score={:.2}, bbox=({}, {}, {}x{}), center=({}, {})",
                    face.score(),
                    bbox.x(),
                    bbox.y(),
                    bbox.width(),
                    bbox.height(),
                    bbox.x() + bbox.width() as i32 / 2,
                    bbox.y() + bbox.height() as i32 / 2,
                );
            }

            // Show what crop the best face would produce
            let best = faces
                .iter()
                .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap())
                .unwrap();
            let bbox = best.bbox();
            let face_cx = bbox.x() as f64 + bbox.width() as f64 / 2.0;
            let face_cy = bbox.y() as f64 + bbox.height() as f64 / 2.0;

            let portrait_aspect = 3.0 / 4.0;
            let (crop_w, crop_h) = if (width as f64 / height as f64) > portrait_aspect {
                let h = height;
                let w = (h as f64 * portrait_aspect).round() as u32;
                (w, h)
            } else {
                let w = width;
                let h = (w as f64 / portrait_aspect).round() as u32;
                (w, h)
            };

            let x = (face_cx - crop_w as f64 / 2.0)
                .round()
                .max(0.0)
                .min((width - crop_w) as f64) as u32;
            let y = (face_cy - crop_h as f64 / 2.0)
                .round()
                .max(0.0)
                .min((height - crop_h) as f64) as u32;

            println!("  → face-centered crop: ({x}, {y}, {crop_w}x{crop_h})");
        }

        // Heuristic crop for comparison
        let portrait_aspect = 3.0 / 4.0;
        let (crop_w, crop_h) = if (width as f64 / height as f64) > portrait_aspect {
            let h = height;
            let w = (h as f64 * portrait_aspect).round() as u32;
            (w, h)
        } else {
            let w = width;
            let h = (w as f64 / portrait_aspect).round() as u32;
            (w, h)
        };
        let x = (width - crop_w) / 2;
        let slack = height - crop_h;
        let y = (slack as f64 * 0.2).round() as u32;
        println!("  → heuristic crop:     ({x}, {y}, {crop_w}x{crop_h})");
        println!();
    }
}
