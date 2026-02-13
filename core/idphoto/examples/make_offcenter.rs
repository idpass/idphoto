//! Generate off-center versions of sample images for testing face detection.
//!
//! Adds canvas padding to shift the face away from center.
//!
//! Usage:
//!   cargo run --example make_offcenter

use image::{DynamicImage, RgbImage};

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures");

fn pad_image(img: &DynamicImage, left: u32, top: u32, right: u32, bottom: u32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let new_w = w + left + right;
    let new_h = h + top + bottom;
    let mut canvas = RgbImage::from_pixel(new_w, new_h, image::Rgb([220, 220, 220]));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        canvas.put_pixel(x + left, y + top, *pixel);
    }

    DynamicImage::ImageRgb8(canvas)
}

fn main() {
    let output_dir = format!("{FIXTURE_DIR}/offcenter");
    std::fs::create_dir_all(&output_dir).unwrap();

    let samples = [
        "sample_id_1.png",
        "sample_id_2.png",
        "sample_id_3.png",
        "sample_id_4.png",
    ];

    for sample in &samples {
        let input_path = format!("{FIXTURE_DIR}/{sample}");
        let img = image::open(&input_path).unwrap();
        let stem = std::path::Path::new(sample)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        // Face pushed to the right: add 400px padding on the left
        let right_shifted = pad_image(&img, 400, 0, 0, 0);
        let path = format!("{output_dir}/{stem}_right.png");
        right_shifted.save(&path).unwrap();
        println!(
            "Created {path} ({}x{})",
            right_shifted.width(),
            right_shifted.height()
        );

        // Face pushed to bottom-right: add padding on left and top
        let bottom_right = pad_image(&img, 300, 300, 100, 0);
        let path = format!("{output_dir}/{stem}_bottomright.png");
        bottom_right.save(&path).unwrap();
        println!(
            "Created {path} ({}x{})",
            bottom_right.width(),
            bottom_right.height()
        );

        // Face in upper-left of a wide image: add padding on right and bottom
        let wide = pad_image(&img, 0, 0, 500, 200);
        let path = format!("{output_dir}/{stem}_wide.png");
        wide.save(&path).unwrap();
        println!("Created {path} ({}x{})", wide.width(), wide.height());
    }

    println!("\nOff-center images written to {output_dir}");
}
