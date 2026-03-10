use std::sync::LazyLock;

use tract_onnx::prelude::*;

use crate::face_detector::{FaceBounds, FaceDetector, FaceLandmarks};

/// Input size the YuNet model was trained on (height and width).
const MODEL_SIZE: usize = 640;

/// Score threshold for filtering low-confidence detections.
const SCORE_THRESHOLD: f32 = 0.5;

/// IoU threshold for non-maximum suppression.
const NMS_IOU_THRESHOLD: f32 = 0.3;

/// Strides used by YuNet's three feature map heads.
const STRIDES: [usize; 3] = [8, 16, 32];

type YunetModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Parsed YuNet ONNX model, loaded once from the bundled binary data.
static YUNET_MODEL: LazyLock<YunetModel> = LazyLock::new(|| {
    let model_bytes: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../model/face_detection_yunet_2023mar.onnx"
    ));

    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(model_bytes))
        .expect("failed to parse YuNet ONNX model")
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![1usize, 3usize, MODEL_SIZE, MODEL_SIZE],
            ),
        )
        .expect("failed to set YuNet input shape")
        .into_optimized()
        .expect("failed to optimize YuNet model")
        .into_runnable()
        .expect("failed to make YuNet model runnable")
});

/// An intermediate detection candidate before NMS.
#[derive(Clone)]
struct Detection {
    /// Bounding box: (x, y, width, height) in original image coordinates.
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    /// Detection score (sigmoid(cls) * sigmoid(obj)).
    score: f32,
    /// 5-point landmarks in original image coordinates [(x0,y0), …, (x4,y4)].
    landmarks: [(f32, f32); 5],
}

/// Compute the intersection-over-union of two detections.
fn iou(a: &Detection, b: &Detection) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;

    let inter_x1 = a.x.max(b.x);
    let inter_y1 = a.y.max(b.y);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Apply greedy non-maximum suppression, returning the kept detections.
///
/// Detections are sorted by score descending. Any detection whose IoU with
/// a previously kept detection exceeds `iou_threshold` is suppressed.
fn nms(detections: &mut [Detection], iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&detections[i], &detections[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Face detector backed by the YuNet ONNX model via `tract-onnx`.
///
/// The bundled YuNet model is parsed once (on first use) and cached in a
/// process-wide static. Construction is free after the first call.
///
/// YuNet produces 5-point facial landmarks (right eye, left eye, nose tip,
/// right mouth corner, left mouth corner) alongside the bounding box.
pub struct YunetDetector;

impl YunetDetector {
    /// Create a new YuNet detector.
    pub fn new() -> Self {
        Self
    }
}

impl Default for YunetDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetector for YunetDetector {
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
        let orig_w = width as usize;
        let orig_h = height as usize;

        // YuNet expects BGR float32 NCHW [1, 3, 640, 640].
        // The grayscale input is expanded to a 3-channel BGR tensor by
        // replicating the single channel across B, G, and R.
        let mut input = vec![0f32; 3 * MODEL_SIZE * MODEL_SIZE];

        let scale_x = orig_w as f32 / MODEL_SIZE as f32;
        let scale_y = orig_h as f32 / MODEL_SIZE as f32;

        // Sample the grayscale image at model-space coordinates using
        // nearest-neighbor interpolation.
        for row in 0..MODEL_SIZE {
            for col in 0..MODEL_SIZE {
                // Map model pixel back to source pixel.
                let src_col = ((col as f32 * scale_x) as usize).min(orig_w - 1);
                let src_row = ((row as f32 * scale_y) as usize).min(orig_h - 1);
                let gray_val = gray[src_row * orig_w + src_col] as f32;

                let pixel_idx = row * MODEL_SIZE + col;
                // B channel
                input[pixel_idx] = gray_val;
                // G channel
                input[MODEL_SIZE * MODEL_SIZE + pixel_idx] = gray_val;
                // R channel
                input[2 * MODEL_SIZE * MODEL_SIZE + pixel_idx] = gray_val;
            }
        }

        let tensor: Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, MODEL_SIZE, MODEL_SIZE), input)
                .expect("shape always matches")
                .into();

        let plan = &*YUNET_MODEL;
        let outputs = match plan.run(tvec![tensor.into()]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("YuNet inference error: {e}");
                return vec![];
            }
        };

        // Output order from the YuNet ONNX model (12 tensors):
        //   0: cls_8    [1, N8,  1]   stride=8
        //   1: cls_16   [1, N16, 1]   stride=16
        //   2: cls_32   [1, N32, 1]   stride=32
        //   3: obj_8    [1, N8,  1]
        //   4: obj_16   [1, N16, 1]
        //   5: obj_32   [1, N32, 1]
        //   6: bbox_8   [1, N8,  4]
        //   7: bbox_16  [1, N16, 4]
        //   8: bbox_32  [1, N32, 4]
        //   9: kps_8    [1, N8,  10]
        //  10: kps_16   [1, N16, 10]
        //  11: kps_32   [1, N32, 10]
        //
        // Where N_s = (MODEL_SIZE / s)^2.
        let mut all_detections: Vec<Detection> = Vec::new();

        for (stride_idx, &stride) in STRIDES.iter().enumerate() {
            let feat = MODEL_SIZE / stride;
            let n = feat * feat;

            let cls_tensor = outputs[stride_idx]
                .to_array_view::<f32>()
                .expect("cls output must be f32");
            let obj_tensor = outputs[3 + stride_idx]
                .to_array_view::<f32>()
                .expect("obj output must be f32");
            let bbox_tensor = outputs[6 + stride_idx]
                .to_array_view::<f32>()
                .expect("bbox output must be f32");
            let kps_tensor = outputs[9 + stride_idx]
                .to_array_view::<f32>()
                .expect("kps output must be f32");

            // Flatten to 1-D slices for index arithmetic.
            let cls_slice = cls_tensor.as_slice().expect("cls is contiguous");
            let obj_slice = obj_tensor.as_slice().expect("obj is contiguous");
            let bbox_slice = bbox_tensor.as_slice().expect("bbox is contiguous");
            let kps_slice = kps_tensor.as_slice().expect("kps is contiguous");

            for cell in 0..n {
                let score = sigmoid(cls_slice[cell]) * sigmoid(obj_slice[cell]);
                if score < SCORE_THRESHOLD {
                    continue;
                }

                let row = cell / feat;
                let col = cell % feat;
                let anchor_cx = (col as f32 + 0.5) * stride as f32;
                let anchor_cy = (row as f32 + 0.5) * stride as f32;

                // Decode bounding box offsets.
                let dx = bbox_slice[cell * 4];
                let dy = bbox_slice[cell * 4 + 1];
                let dw = bbox_slice[cell * 4 + 2];
                let dh = bbox_slice[cell * 4 + 3];

                let cx = anchor_cx + dx * stride as f32;
                let cy = anchor_cy + dy * stride as f32;
                let bw = dw.exp() * stride as f32;
                let bh = dh.exp() * stride as f32;

                // Decode 5 landmark offsets.
                let mut landmarks = [(0f32, 0f32); 5];
                for j in 0..5 {
                    let lx = anchor_cx + kps_slice[cell * 10 + j * 2] * stride as f32;
                    let ly = anchor_cy + kps_slice[cell * 10 + j * 2 + 1] * stride as f32;
                    landmarks[j] = (lx, ly);
                }

                // Scale back from model space to original image coordinates.
                let scale_fx = orig_w as f32 / MODEL_SIZE as f32;
                let scale_fy = orig_h as f32 / MODEL_SIZE as f32;

                let final_cx = cx * scale_fx;
                let final_cy = cy * scale_fy;
                let final_w = bw * scale_fx;
                let final_h = bh * scale_fy;
                let final_x = final_cx - final_w / 2.0;
                let final_y = final_cy - final_h / 2.0;

                let scaled_landmarks = landmarks.map(|(lx, ly)| (lx * scale_fx, ly * scale_fy));

                all_detections.push(Detection {
                    x: final_x,
                    y: final_y,
                    w: final_w,
                    h: final_h,
                    score,
                    landmarks: scaled_landmarks,
                });
            }
        }

        let kept = nms(&mut all_detections, NMS_IOU_THRESHOLD);

        kept.into_iter()
            .map(|det| {
                let lm = det.landmarks;
                FaceBounds {
                    x: det.x as f64,
                    y: det.y as f64,
                    width: det.w as f64,
                    height: det.h as f64,
                    confidence: det.score as f64,
                    landmarks: Some(FaceLandmarks {
                        right_eye: (lm[0].0 as f64, lm[0].1 as f64),
                        left_eye: (lm[1].0 as f64, lm[1].1 as f64),
                        nose: (lm[2].0 as f64, lm[2].1 as f64),
                        right_mouth: (lm[3].0 as f64, lm[3].1 as f64),
                        left_mouth: (lm[4].0 as f64, lm[4].1 as f64),
                    }),
                }
            })
            .collect()
    }
}
