//! Generate all preset versions for every sample image.
//!
//! Usage:
//!   cargo run --example generate_samples [--features face-detection]
//!
//! Output goes to `tests/fixtures/output/`.

use idphoto::{CropMode, OutputFormat, PhotoCompressor, Preset};
use std::path::Path;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures");

fn process_image(_sample: &str, input: &[u8], source_tag: &str, output_dir: &Path) {
    let presets: &[(&str, Preset)] = &[
        ("qr", Preset::QrCode),
        ("qr_match", Preset::QrCodeMatch),
        ("print", Preset::Print),
        ("display", Preset::Display),
    ];

    for (preset_name, preset) in presets {
        let result = PhotoCompressor::new(input.to_vec())
            .unwrap()
            .preset(preset.clone())
            .compress()
            .unwrap();

        let ext = match result.format {
            OutputFormat::Webp => "webp",
            OutputFormat::Jpeg => "jpg",
        };

        let filename = format!("{source_tag}_{preset_name}.{ext}");
        let output_path = output_dir.join(&filename);
        std::fs::write(&output_path, &result.data).unwrap();

        println!(
            "  {preset_name}: {filename} ({width}x{height}, {size} bytes)",
            width = result.width,
            height = result.height,
            size = result.data.len(),
        );
    }

    // Also generate a heuristic-crop QR for comparison
    let heuristic_qr = PhotoCompressor::new(input.to_vec())
        .unwrap()
        .preset(Preset::QrCode)
        .crop_mode(CropMode::Heuristic)
        .compress()
        .unwrap();

    let filename = format!("{source_tag}_qr_heuristic.webp");
    let output_path = output_dir.join(&filename);
    std::fs::write(&output_path, &heuristic_qr.data).unwrap();

    println!(
        "  qr_heuristic: {filename} ({width}x{height}, {size} bytes)",
        width = heuristic_qr.width,
        height = heuristic_qr.height,
        size = heuristic_qr.data.len(),
    );
}

fn main() {
    let output_dir = Path::new(FIXTURE_DIR).join("output");
    std::fs::create_dir_all(&output_dir).expect("failed to create output directory");

    // Original centered samples
    let samples = [
        "sample_id_1.png",
        "sample_id_2.png",
        "sample_id_3.png",
        "sample_id_3_crop.jpg",
        "sample_id_4.png",
    ];

    for sample in &samples {
        let input_path = format!("{FIXTURE_DIR}/{sample}");
        let input = std::fs::read(&input_path)
            .unwrap_or_else(|e| panic!("failed to read {input_path}: {e}"));

        let stem = Path::new(sample).file_stem().unwrap().to_str().unwrap();
        println!("=== {sample} ===");
        process_image(sample, &input, stem, &output_dir);
        println!();
    }

    // Off-center samples (if they exist)
    let offcenter_dir = format!("{FIXTURE_DIR}/offcenter");
    if Path::new(&offcenter_dir).exists() {
        let offcenter_samples: Vec<_> = std::fs::read_dir(&offcenter_dir)
            .unwrap()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name().into_string().ok()?;
                if name.ends_with(".png") || name.ends_with(".jpg") {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        let mut offcenter_samples = offcenter_samples;
        offcenter_samples.sort();

        for sample in &offcenter_samples {
            let input_path = format!("{offcenter_dir}/{sample}");
            let input = std::fs::read(&input_path).unwrap();
            let stem = Path::new(sample).file_stem().unwrap().to_str().unwrap();
            println!("=== offcenter/{sample} ===");
            process_image(sample, &input, &format!("offcenter_{stem}"), &output_dir);
            println!();
        }
    }

    println!("Output written to {}", output_dir.display());
}
