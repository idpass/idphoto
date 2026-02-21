# idphoto

Identity photo processing library: crop, resize, and compress photos for ID documents and QR codes.

Built in Rust with SDKs for Python, TypeScript/JavaScript, Kotlin, and Java.

## Features

- **Face-aware cropping** — detects faces and frames them in a 3:4 portrait crop with configurable margins (ID-photo framing or tight face crop for matching)
- **Preset configurations** — `QrCode`, `QrCodeMatch`, `Print`, and `Display` presets for common use cases
- **Byte-budget compression** — binary search over quality to fit within a target size (e.g., QR code payload limits)
- **Multiple output formats** — WebP and JPEG (both quality-controlled), with ICC stripping for WebP
- **Grayscale conversion** — smaller payloads for QR workflows (JPEG is encoded single-channel in grayscale mode)
- **Pluggable face detection** — built-in SeetaFace backend (~1.2MB model bundled) or bring your own via the `FaceDetector` trait
- **Safe Rust API** — memory-safe interfaces across core and bindings

## API By Language

- **Rust**: fluent builder (`PhotoCompressor`) + enums (`Preset`, `CropMode`, `OutputFormat`)
- **Python**: enums + typed options (`CompressOptions`, `CompressToFitOptions`) + `IdPhoto` object-style entrypoint
- **TypeScript/JavaScript**: `createIdPhoto()` client API + exported constants (`Preset`, `CropMode`, `OutputFormat`)
- **Kotlin**: DSL-style wrapper (`IdPhoto.compress { ... }`) over UniFFI/JNI bindings
- **Java**: static facade (`JavaIdPhoto.compress(...)`) with Java builders and Java-friendly result models

## Presets

| Preset | Dimension | Format | Grayscale | Crop | Use Case |
|--------|-----------|--------|-----------|------|----------|
| `QrCode` | 48px | WebP | Yes | Face detection | Visual verification in QR codes |
| `QrCodeMatch` | 48px | WebP | Yes | Face detection (tight) | Algorithmic face matching in QR codes |
| `Print` | 400px | JPEG | No | Face detection | Printed ID cards and documents |
| `Display` | 200px | JPEG | No | Face detection | On-screen display and digital verification |

## Quick Start

### Rust

```rust
use idphoto::PhotoCompressor;

let raw_bytes = std::fs::read("photo.jpg").unwrap();
let result = PhotoCompressor::new(raw_bytes)
    .unwrap()
    .max_dimension(48)
    .quality(0.6)
    .compress()
    .unwrap();
println!("Compressed: {} bytes", result.data.len());
```

Using a preset:

```rust
use idphoto::{PhotoCompressor, Preset};

let bytes = std::fs::read("photo.jpg").unwrap();
let result = PhotoCompressor::new(bytes)
    .unwrap()
    .preset(Preset::QrCode)
    .compress()
    .unwrap();
```

Compressing to fit within a byte budget:

```rust
use idphoto::PhotoCompressor;

let bytes = std::fs::read("photo.jpg").unwrap();
let fit = PhotoCompressor::new(bytes)
    .unwrap()
    .max_dimension(48)
    .compress_to_fit(2048) // 2KB budget
    .unwrap();
println!("Quality: {}, Reached target: {}", fit.quality_used, fit.reached_target);
```

### Python

```python
import idphoto

raw = open("photo.jpg", "rb").read()

# Typed options + object-style API
options = idphoto.CompressOptions(preset=idphoto.Preset.QR_CODE)
result = idphoto.IdPhoto.compress(raw, options=options)
print(f"{result.width}x{result.height}, {len(result.data)} bytes")

# With a byte budget
fit = idphoto.IdPhoto.compress_to_fit(
    raw,
    max_bytes=2048,
    options=idphoto.CompressToFitOptions(preset=idphoto.Preset.QR_CODE),
)
print(f"Quality: {fit.quality_used}, Reached: {fit.reached_target}")
```

### TypeScript/JavaScript

Implementation detail: packaged as a WASM module.

```javascript
import { createIdPhoto, Preset } from "@idpass/idphoto-wasm";

const input = new Uint8Array(await file.arrayBuffer());
const idphoto = await createIdPhoto();

const result = idphoto.compress(input, { preset: Preset.QR_CODE });
console.log(`${result.width}x${result.height}, ${result.data.length} bytes`);

const fit = idphoto.compressToFit(input, 2048, { preset: Preset.QR_CODE });
console.log(`Quality: ${fit.qualityUsed}, Reached: ${fit.reachedTarget}`);
```

### Kotlin

```kotlin
import org.idpass.idphoto.*

val result = IdPhoto.compress(rawBytes) {
    preset = Preset.QR_CODE
}
println(result.toSummaryString())
```

### Java

```java
import org.idpass.idphoto.*;

JavaCompressOptions options = new JavaCompressOptions.Builder()
    .setPreset(Preset.QR_CODE)
    .build();
JavaCompressedPhoto result = JavaIdPhoto.compress(rawBytes, options);
System.out.println(JavaIdPhoto.summary(result));
```

## Building from Source

### Prerequisites

- Rust 1.75+ (MSRV)
- Git LFS (for model and test fixture files)

```sh
git lfs install
git clone https://github.com/idpass/idphoto.git
cd idphoto
```

### Core Library

```sh
# Without face detection
cargo build -p idphoto

# With built-in SeetaFace face detection
cargo build -p idphoto --features rustface
```

### Python Bindings

```sh
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

### TypeScript/JavaScript Binding (WASM)

```sh
rustup target add wasm32-unknown-unknown
cargo build -p idphoto-wasm --target wasm32-unknown-unknown --release
```

### Kotlin And Java Bindings (UniFFI/JNI)

```sh
cargo build -p idphoto-jni
```

## Running Tests

```sh
# All Rust tests (core + bindings)
cargo test --workspace --all-features

# Python tests
pip install maturin pytest
maturin develop --manifest-path bindings/python/Cargo.toml
pytest bindings/python/tests/ -v
```

## Documentation Site

```sh
python3 -m pip install -r docs/requirements.txt
mkdocs serve
```

Build static output:

```sh
mkdocs build --strict
```

## Custom Face Detector

Implement the `FaceDetector` trait to plug in your own detection backend:

```rust
use idphoto::{PhotoCompressor, FaceDetector, FaceBounds};

struct MyDetector;

impl FaceDetector for MyDetector {
    fn detect(&self, gray: &[u8], width: u32, height: u32) -> Vec<FaceBounds> {
        // Your detection logic (ONNX, dlib, etc.)
        vec![]
    }
}

let bytes = std::fs::read("photo.jpg").unwrap();
let result = PhotoCompressor::new(bytes)
    .unwrap()
    .face_detector(Box::new(MyDetector))
    .compress()
    .unwrap();
```

## Architecture

```
idphoto/
├── core/idphoto/       # Core Rust library
│   └── src/
│       ├── lib.rs              # Public API (PhotoCompressor, presets)
│       ├── compress.rs         # Compression pipeline
│       ├── crop.rs             # Heuristic portrait cropping
│       ├── face_detector.rs    # FaceDetector trait + FaceBounds
│       ├── rustface_backend.rs # Built-in SeetaFace backend
│       ├── error.rs            # Error types
│       └── webp_strip.rs       # ICC profile stripping
├── bindings/
│   ├── python/         # PyO3 bindings
│   ├── wasm/           # wasm-bindgen bindings
│   └── jni/            # UniFFI bindings for Android/JVM
├── model/              # Bundled SeetaFace model (Git LFS)
└── tests/fixtures/     # Synthetic test images (Git LFS)
```

## License

MIT — see [LICENSE](LICENSE) for details.

The bundled SeetaFace model (`model/seeta_fd_frontal_v1.0.bin`) is licensed under BSD 2-Clause by Seetatech. See [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) for details.
