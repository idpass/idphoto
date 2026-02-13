/// Portrait aspect ratio: 3:4 (width / height)
const PORTRAIT_ASPECT: f64 = 3.0 / 4.0;

/// Vertical bias toward the top of the image (faces in upper portion).
/// 0.0 = top, 0.5 = center, 1.0 = bottom.
const VERTICAL_BIAS: f64 = 0.2;

/// Crop region within the source image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Calculate the 3:4 portrait crop region for the given source dimensions.
///
/// Centers horizontally and biases toward the top vertically (20% from top)
/// to capture faces in typical portrait photos.
pub fn portrait_crop(source_width: u32, source_height: u32) -> CropRegion {
    let (crop_width, crop_height) =
        if (source_width as f64 / source_height as f64) > PORTRAIT_ASPECT {
            // Source is wider than 3:4 — constrain by height
            let h = source_height;
            let w = (h as f64 * PORTRAIT_ASPECT).round() as u32;
            (w, h)
        } else {
            // Source is taller than (or equal to) 3:4 — constrain by width
            let w = source_width;
            let h = (w as f64 / PORTRAIT_ASPECT).round() as u32;
            (w, h)
        };

    // Center horizontally
    let x = (source_width.saturating_sub(crop_width)) / 2;

    // Bias toward top vertically
    let vertical_slack = source_height.saturating_sub(crop_height);
    let y = (vertical_slack as f64 * VERTICAL_BIAS).round() as u32;

    CropRegion {
        x,
        y,
        width: crop_width,
        height: crop_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_source_constrains_by_width() {
        // 100x100 is wider than 3:4, so constrain by... actually 1:1 > 3:4
        // width/height = 1.0 > 0.75, so constrain by height
        let crop = portrait_crop(100, 100);
        assert_eq!(crop.width, 75); // 100 * 0.75
        assert_eq!(crop.height, 100);
        assert_eq!(crop.x, 12); // (100 - 75) / 2 = 12
        assert_eq!(crop.y, 0); // no vertical slack
    }

    #[test]
    fn tall_source_constrains_by_width() {
        // 300x800 — aspect 0.375 < 0.75, so constrain by width
        let crop = portrait_crop(300, 800);
        assert_eq!(crop.width, 300);
        assert_eq!(crop.height, 400); // 300 / 0.75
        assert_eq!(crop.x, 0);
        // Vertical slack = 800 - 400 = 400, bias 20% → y = 80
        assert_eq!(crop.y, 80);
    }

    #[test]
    fn wide_source_constrains_by_height() {
        // 800x300 — aspect 2.67 > 0.75, so constrain by height
        let crop = portrait_crop(800, 300);
        assert_eq!(crop.width, 225); // 300 * 0.75
        assert_eq!(crop.height, 300);
        assert_eq!(crop.x, 287); // (800 - 225) / 2
        assert_eq!(crop.y, 0);
    }

    #[test]
    fn exact_3_4_ratio_no_crop_needed() {
        let crop = portrait_crop(300, 400);
        assert_eq!(crop.width, 300);
        assert_eq!(crop.height, 400);
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
    }

    #[test]
    fn small_source() {
        let crop = portrait_crop(3, 4);
        assert_eq!(crop.width, 3);
        assert_eq!(crop.height, 4);
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
    }

    #[test]
    fn vertical_bias_pushes_crop_toward_top() {
        // 100x1000, very tall — constrain by width
        let crop = portrait_crop(100, 1000);
        assert_eq!(crop.width, 100);
        assert_eq!(crop.height, 133); // 100 / 0.75 ≈ 133
                                      // Vertical slack = 1000 - 133 = 867, bias 20% → y ≈ 173
        let expected_y = (867.0_f64 * 0.2).round() as u32;
        assert_eq!(crop.y, expected_y);
    }
}
