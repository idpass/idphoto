# Cropping Modes

`idphoto` supports three crop strategies:

| Mode | Behavior | Good For |
| --- | --- | --- |
| `FaceDetection` (default) | Detects best face, crops around it, falls back to heuristic if no face found | Most production flows |
| `Heuristic` | 3:4 portrait crop with upper vertical bias | Inputs without reliable face detection |
| `None` | No crop, only resize | Preserve original framing |

## Heuristic Crop

The heuristic path targets a `3:4` portrait aspect ratio and biases the crop upward (`20%` from top) to keep typical head placement in frame.

## Face Detection Crop

With `FaceDetection`, the crop is sized from detected face height using:

`crop_height = face_height * face_margin`

- `face_margin=2.0` gives ID framing (hair + shoulders)
- `face_margin=1.3` gives tighter framing for matching

If detection fails, `idphoto` falls back automatically to the heuristic crop.

## Example

```rust
use idphoto::{CropMode, PhotoCompressor};

let bytes = std::fs::read("photo.jpg")?;
let output = PhotoCompressor::new(bytes)?
    .crop_mode(CropMode::FaceDetection)
    .face_margin(2.0)
    .compress()?;

println!("{}x{}", output.width, output.height);
# Ok::<(), idphoto::IdPhotoError>(())
```
