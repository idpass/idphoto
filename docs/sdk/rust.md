# Rust SDK

Crate: `idphoto` (`core/idphoto`)

## Install

`idphoto` is not published on crates.io yet. Use a Git dependency:

```toml
[dependencies]
idphoto = { git = "https://github.com/idpass/idphoto", package = "idphoto" }
```

Or use path dependency while developing:

```toml
[dependencies]
idphoto = { path = "core/idphoto" }
```

## Basic Usage

```rust
use idphoto::{PhotoCompressor, Preset};

let input = std::fs::read("photo.jpg")?;
let result = PhotoCompressor::new(input)?
    .preset(Preset::Display)
    .compress()?;

println!("{}x{}", result.width, result.height);
# Ok::<(), idphoto::IdPhotoError>(())
```

## Byte Budget

```rust
use idphoto::{PhotoCompressor, Preset};

let input = std::fs::read("photo.jpg")?;
let fit = PhotoCompressor::new(input)?
    .preset(Preset::QrCode)
    .compress_to_fit(2048)?;

println!("quality={}, reached={}", fit.quality_used, fit.reached_target);
# Ok::<(), idphoto::IdPhotoError>(())
```

## Builder Options

| Method | Purpose |
| --- | --- |
| `preset(Preset)` | Apply complete preset defaults |
| `max_dimension(u32)` | Set output max dimension |
| `quality(f32)` | Set encoding quality (`0.0..=1.0`) for JPEG and lossy WebP |
| `grayscale(bool)` | Enable grayscale conversion |
| `crop_mode(CropMode)` | Set crop strategy |
| `format(OutputFormat)` | Select `Webp` or `Jpeg` |
| `face_margin(f32)` | Adjust face crop framing |
| `face_detector(Box<dyn FaceDetector>)` | Plug in custom detector |
