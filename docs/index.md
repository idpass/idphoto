<div class="hero-section">
  <p class="hero-title">idphoto</p>
  <p>Identity photo processing for QR payloads, digital display, and print workflows.</p>
  <p><strong>Crop</strong> -> <strong>resize</strong> -> <strong>compress</strong>, with preset defaults tuned for real deployment constraints.</p>
</div>

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
    <p>Rust core with Python, JavaScript (WASM), and Kotlin/Java (JNI/UniFFI) bindings.</p>
  </div>
</div>

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

## Quick Start (Rust)

```rust
use idphoto::{PhotoCompressor, Preset};

let bytes = std::fs::read("photo.jpg")?;
let result = PhotoCompressor::new(bytes)?
    .preset(Preset::QrCode)
    .compress()?;

println!("{}x{}, {} bytes", result.width, result.height, result.data.len());
# Ok::<(), idphoto::IdPhotoError>(())
```

## Next Steps

- Start with [Getting Started](getting-started.md)
- Understand [Processing Pipeline](core/pipeline.md)
- Choose defaults from [Presets and Byte Budgets](core/presets.md)
- Integrate using [SDK docs](sdk/rust.md)
