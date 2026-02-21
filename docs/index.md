<div class="hero-section">
  <p class="hero-title">idphoto</p>
  <p>Identity photo processing for QR payloads, digital display, and print workflows.</p>
  <p><strong>Crop</strong> -> <strong>resize</strong> -> <strong>compress</strong>, with preset defaults tuned for real deployment constraints.</p>
</div>

## Start Here

1. [Getting Started](getting-started.md) for prerequisites and local setup.
2. [SDK Overview](sdk/index.md) to pick your language-specific API.
3. [Presets and Byte Budgets](core/presets.md) to choose default behavior.

## Why idphoto

<div class="feature-grid">
  <div class="feature-card">
    <h3>Face-aware framing</h3>
    <p>Face detection crop with fallback to heuristic 3:4 portrait framing for robust behavior.</p>
  </div>
  <div class="feature-card">
    <h3>Preset-driven</h3>
    <p>Use `QrCode`, `QrCodeMatch`, `Display`, and `Print` presets, then override only what you need.</p>
  </div>
  <div class="feature-card">
    <h3>Byte-budget targeting</h3>
    <p>`compress_to_fit` runs binary search over quality to hit payload size targets.</p>
  </div>
  <div class="feature-card">
    <h3>Cross-platform bindings</h3>
    <p>Rust core with Python, TypeScript/JavaScript, Kotlin, and Java SDKs.</p>
  </div>
</div>

## SDK Entry Points

| Language | Primary API | Details |
| --- | --- | --- |
| Rust | `PhotoCompressor::new(...).preset(...).compress()` | [Rust SDK](sdk/rust.md) |
| Python | `IdPhoto.compress(..., options=CompressOptions(...))` | [Python SDK](sdk/python.md) |
| TypeScript/JavaScript | `const api = await createIdPhoto(); api.compress(...)` | [TypeScript/JavaScript SDK](sdk/wasm.md) |
| Kotlin | `IdPhoto.compress(bytes) { ... }` | [Kotlin SDK](sdk/kotlin.md) |
| Java | `JavaIdPhoto.compress(bytes, options)` | [Java SDK](sdk/java.md) |

## Pipeline At A Glance

<div class="pipeline">
  <span class="pipeline-step">Decode</span>
  <span class="pipeline-arrow">-></span>
  <span class="pipeline-step">Crop (face/heuristic/none)</span>
  <span class="pipeline-arrow">-></span>
  <span class="pipeline-step">Resize</span>
  <span class="pipeline-arrow">-></span>
  <span class="pipeline-step">Flatten alpha</span>
  <span class="pipeline-arrow">-></span>
  <span class="pipeline-step">Grayscale (optional)</span>
  <span class="pipeline-arrow">-></span>
  <span class="pipeline-step">Encode (WebP/JPEG)</span>
</div>

## Quick Example (Rust)

```rust
use idphoto::{PhotoCompressor, Preset};

let bytes = std::fs::read("photo.jpg")?;
let result = PhotoCompressor::new(bytes)?
    .preset(Preset::QrCode)
    .compress()?;

println!("{}x{}, {} bytes", result.width, result.height, result.data.len());
# Ok::<(), idphoto::IdPhotoError>(())
```
