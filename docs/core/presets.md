# Presets And Byte Budgets

Presets set all processing parameters at once. Apply a preset first, then override only fields that need to change.

## Built-In Presets

| Rust Enum | Python/JS String | Max Dimension | Format | Grayscale | Crop Intent |
| --- | --- | --- | --- | --- | --- |
| `QrCode` | `"qr-code"` | `48` | WebP | Yes | Visual verification in QR payloads |
| `QrCodeMatch` | `"qr-code-match"` | `48` | WebP | Yes | Tighter crop for matching |
| `Display` | `"display"` | `200` | JPEG (`0.75`) | No | On-screen display |
| `Print` | `"print"` | `400` | JPEG (`0.9`) | No | Printed cards/documents |

## Overriding A Preset

```rust
use idphoto::{OutputFormat, PhotoCompressor, Preset};

let bytes = std::fs::read("photo.jpg")?;
let result = PhotoCompressor::new(bytes)?
    .preset(Preset::QrCode)
    .format(OutputFormat::Jpeg) // override only format
    .compress()?;

println!("{} bytes", result.data.len());
# Ok::<(), idphoto::IdPhotoError>(())
```

## `compress_to_fit(max_bytes)`

`compress_to_fit` runs binary search over quality (8 iterations) to find the highest quality that stays within `max_bytes`.

- If target is achievable: returns `reached_target = true` and chosen quality.
- If target is impossible even at minimum quality: returns best-effort output with `reached_target = false`.

```rust
use idphoto::{PhotoCompressor, Preset};

let bytes = std::fs::read("photo.jpg")?;
let fit = PhotoCompressor::new(bytes)?
    .preset(Preset::QrCode)
    .compress_to_fit(2048)?;

println!("quality={} reached={}", fit.quality_used, fit.reached_target);
# Ok::<(), idphoto::IdPhotoError>(())
```
