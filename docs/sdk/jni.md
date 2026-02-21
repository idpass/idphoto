# JVM Low-Level API (UniFFI/JNI)

Binding crate: `idphoto-jni`

This page documents the generated UniFFI/JNI layer shared by the Java and Kotlin wrappers.

- Kotlin wrapper: [Kotlin SDK](kotlin.md)
- Java wrapper: [Java SDK](java.md)

## Build

```bash
cargo build -p idphoto-jni
```

## Generated Functions

### `compress(input, options)`

- `input`: `ByteArray`
- `options`: `CompressOptions`
- returns: `CompressedPhoto`

### `compressToFit(input, maxBytes, options)`

- `input`: `ByteArray`
- `maxBytes`: `Long`/`ULong` bridge value
- `options`: `CompressOptions`
- returns: `FitResult`

### `defaultCompressOptions()`

Returns a `CompressOptions` with all fields set to `null`.

## Generated Records And Enums

- `CompressOptions`
- `CompressedPhoto`
- `FitResult`
- `FaceBounds`
- `Preset`, `CropMode`, `OutputFormat`

## Error Mapping

Core `idphoto::IdPhotoError` values are mapped to UniFFI-friendly `IdPhotoError` variants:

- `DecodeError`
- `UnsupportedFormat`
- `ZeroDimensions`
- `EncodeError`
- `InvalidQuality`
- `InvalidMaxDimension`
