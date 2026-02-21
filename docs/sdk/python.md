# Python SDK

Module: `idphoto` (`bindings/python`)

See [SDK Overview](index.md) for language selection and API shape comparison.

## Build And Install Locally

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

## API

Use enums and typed option objects instead of raw string literals:

- `Preset`, `CropMode`, `OutputFormat`
- `CompressOptions`
- `CompressToFitOptions`
- `IdPhoto` object-style entrypoint

```python
import idphoto

raw = open("photo.jpg", "rb").read()

options = idphoto.CompressOptions(
    preset=idphoto.Preset.QR_CODE,
    crop_mode=idphoto.CropMode.FACE_DETECTION,
    output_format=idphoto.OutputFormat.WEBP,
)

result = idphoto.IdPhoto.compress(raw, options=options)
print(result.format, len(result.data))

fit_options = idphoto.CompressToFitOptions(
    preset=idphoto.Preset.QR_CODE,
)
fit = idphoto.IdPhoto.compress_to_fit(raw, 2048, options=fit_options)
print(fit.quality_used, fit.reached_target)
```

## Functional API (Compatible)

The original function-style API remains supported:

```python
result = idphoto.compress(raw, preset="qr-code")
fit = idphoto.compress_to_fit(raw, max_bytes=2048, preset="qr-code")
```

## Return Types

`compress` returns `CompressResult` with attributes:

- `data` (`bytes`) — compressed image bytes
- `format` (`str`) — `"webp"` or `"jpeg"`
- `width` (`int`), `height` (`int`)
- `original_size` (`int`)
- `face_bounds` (`FaceBounds | None`)

`compress_to_fit` returns `FitResult` with all `CompressResult` attributes plus:

- `quality_used` (`float`)
- `reached_target` (`bool`)

## Exception Hierarchy

All library errors inherit from `IdPhotoError`, which inherits from `Exception`:

| Exception | Raised When |
| --- | --- |
| `DecodeError` | Input cannot be decoded as an image |
| `UnsupportedFormatError` | Reserved for unsupported-format mapping from core |
| `ZeroDimensionsError` | Image dimensions are zero |
| `EncodeError` | Output image cannot be encoded |
| `InvalidQualityError` | Quality is outside 0.0–1.0 |
| `InvalidMaxDimensionError` | max_dimension is 0 |

Note: unsupported input formats are currently reported as `DecodeError` by the core decoder path.

Argument validation errors (invalid preset, crop_mode, or output_format strings) raise `ValueError`.

## Type Support

The package ships with a `py.typed` marker and `.pyi` stubs for static type checking.
