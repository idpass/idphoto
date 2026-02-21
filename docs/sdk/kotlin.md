# Kotlin SDK

Package namespace: `org.idpass.idphoto`

Primary wrapper: `IdPhoto` (`bindings/jni/kotlin/IdPhoto.kt`)

## Build

```bash
cargo build -p idphoto-jni
```

## API

Use the DSL wrapper for Kotlin call sites:

- `IdPhoto.compress(data) { ... }`
- `IdPhoto.compressToFit(data, maxBytes) { ... }`

```kotlin
import org.idpass.idphoto.*

val raw = File("photo.jpg").readBytes()

val result = IdPhoto.compress(raw) {
    preset = Preset.QR_CODE
    format = OutputFormat.JPEG
}
println(result.toSummaryString())

val fit = IdPhoto.compressToFit(raw, maxBytes = 2048L) {
    preset = Preset.QR_CODE
}
println(fit.toSummaryString())
```

## Options DSL

The `CompressOptionsBuilder` block supports:

- `preset`
- `maxDimension`
- `quality` (compress only)
- `grayscale`
- `cropMode`
- `format`
- `faceMargin`

## Kotlin Helpers

`IdPhoto.kt` also provides helper extensions:

- `CompressedPhoto.mimeType`
- `CompressedPhoto.fileExtension`
- `CompressedPhoto.toSummaryString()`
- `FitResult.toSummaryString()`

## Advanced

For lower-level generated bindings and type mapping details, see
[JVM Low-Level API (UniFFI/JNI)](jni.md).
