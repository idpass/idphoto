# Face Detection

## Built-In Backend

`idphoto` can use a bundled SeetaFace model backend when compiled with the `rustface` feature.

```bash
cargo build -p idphoto --features rustface
```

This enables automatic detection when `CropMode::FaceDetection` is active and no custom detector is provided.

## Custom Detector Integration

You can plug in a custom detector by implementing the `FaceDetector` trait and passing it through `PhotoCompressor::face_detector`.

```rust
use idphoto::{FaceBounds, FaceDetector, PhotoCompressor};

struct MyDetector;

impl FaceDetector for MyDetector {
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
        let _ = (gray, width, height);
        vec![]
    }
}

let bytes = std::fs::read("photo.jpg")?;
let out = PhotoCompressor::new(bytes)?
    .face_detector(Box::new(MyDetector))
    .compress()?;

println!("{} bytes", out.data.len());
# Ok::<(), idphoto::IdPhotoError>(())
```

## Detector Selection Rules

1. If a custom detector is provided, it is used.
2. Otherwise, if `rustface` feature is enabled, the built-in backend is used.
3. Otherwise, no detection is performed and crop falls back to heuristic behavior.
