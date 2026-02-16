# Kotlin/Java SDK (JNI)

Binding crate: `idphoto-jni` (`bindings/jni`)

The JNI layer is exported with UniFFI records/enums and is intended for Android/JVM integration.

## Build

```bash
cargo build -p idphoto-jni
```

## API

### `compress(input, options)`

Compress a photo with the given options.

- `input` — `ByteArray` of raw image bytes (JPEG, PNG, or WebP)
- `options` — `CompressOptions` record

Returns a `CompressedPhoto` record.

### `compressToFit(input, maxBytes, options)`

Compress to fit within a byte budget.

- `input` — `ByteArray` of raw image bytes
- `maxBytes` — `Long`, maximum output size in bytes
- `options` — `CompressOptions` record

Returns a `FitResult` record.

### `defaultCompressOptions()`

Returns a `CompressOptions` with all fields set to `null`, suitable as a starting point for customisation.

## CompressOptions

All fields are optional (`null` means "use default" or "use preset value"):

| Field | Type | Description |
| --- | --- | --- |
| `preset` | `Preset?` | `QrCode`, `QrCodeMatch`, `Print`, or `Display` |
| `maxDimension` | `UInt?` | Maximum output dimension in pixels |
| `quality` | `Float?` | Compression quality 0.0–1.0 |
| `grayscale` | `Boolean?` | Convert to grayscale |
| `cropMode` | `CropMode?` | `Heuristic`, `FaceDetection`, or `None` |
| `format` | `OutputFormat?` | `Webp` or `Jpeg` |
| `faceMargin` | `Float?` | Face detection crop margin multiplier |

## Data Models

- `CompressedPhoto`: `data` (`ByteArray`), `format` (`OutputFormat`), `width` (`UInt`), `height` (`UInt`), `originalSize` (`ULong`), `faceBounds` (`FaceBounds?`)
- `FitResult`: `photo` (`CompressedPhoto`), `qualityUsed` (`Float`), `reachedTarget` (`Boolean`)
- `FaceBounds`: `x`, `y`, `width`, `height`, `confidence` (all `Double`)

## Error Handling

Core `idphoto::IdPhotoError` values are mapped to UniFFI-friendly `IdPhotoError` variants for language-safe handling in Kotlin/Java:

| Variant | Description |
| --- | --- |
| `DecodeError` | Input cannot be decoded (carries message) |
| `UnsupportedFormat` | Input is not JPEG, PNG, or WebP |
| `ZeroDimensions` | Image dimensions are zero |
| `EncodeError` | Output encoding failed (carries message) |
| `InvalidQuality` | Quality outside 0.0–1.0 (carries the invalid value) |
| `InvalidMaxDimension` | maxDimension is 0 |

## Example (Kotlin)

```kotlin
import idphoto.*

val raw = File("photo.jpg").readBytes()

// Using default options
val result = compress(raw, defaultCompressOptions())
println("${result.width}x${result.height}, ${result.data.size} bytes")

// With a preset
val options = defaultCompressOptions().copy(preset = Preset.QR_CODE)
val result2 = compress(raw, options)

// Preset with overrides
val options2 = defaultCompressOptions().copy(
    preset = Preset.QR_CODE,
    format = OutputFormat.JPEG,
)
val result3 = compress(raw, options2)

// Byte budget
val fit = compressToFit(raw, 2048L, defaultCompressOptions().copy(
    preset = Preset.QR_CODE,
))
println("quality=${fit.qualityUsed}, reached=${fit.reachedTarget}")
```

## Kotlin Wrapper

A convenience Kotlin wrapper (`bindings/jni/kotlin/IdPhoto.kt`) provides a DSL-style API with default parameter values:

```kotlin
val result = IdPhoto.compress(raw) {
    preset = Preset.QR_CODE
    format = OutputFormat.JPEG
}

val fit = IdPhoto.compressToFit(raw, maxBytes = 2048L) {
    preset = Preset.QR_CODE
}
```
