# Python SDK

Module: `idphoto` (`bindings/python`)

## Build And Install Locally

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

## API

### `compress(data, *, ...)`

Compress an identity photo.

```python
compress(
    data: bytes,
    *,
    preset: str | None = None,
    max_dimension: int | None = None,
    quality: float | None = None,
    grayscale: bool | None = None,
    crop_mode: str | None = None,
    output_format: str | None = None,
    face_margin: float | None = None,
) -> CompressResult
```

Returns a `CompressResult` with attributes:

- `data` (`bytes`) — compressed image bytes
- `format` (`str`) — `"webp"` or `"jpeg"`
- `width` (`int`), `height` (`int`)
- `original_size` (`int`)
- `face_bounds` (`FaceBounds | None`)

### `compress_to_fit(data, max_bytes, *, ...)`

Compress to fit within a byte budget. Uses binary search over quality (8 iterations).

```python
compress_to_fit(
    data: bytes,
    max_bytes: int,
    *,
    preset: str | None = None,
    max_dimension: int | None = None,
    grayscale: bool | None = None,
    crop_mode: str | None = None,
    output_format: str | None = None,
    face_margin: float | None = None,
) -> FitResult
```

Returns a `FitResult` with all `CompressResult` attributes plus:

- `quality_used` (`float`)
- `reached_target` (`bool`)

### `FaceBounds`

- `x` (`float`), `y` (`float`), `width` (`float`), `height` (`float`)
- `confidence` (`float`)

## Example

```python
import idphoto

raw = open("photo.jpg", "rb").read()

result = idphoto.compress(raw, preset="qr-code")
print(result.format, len(result.data))

fit = idphoto.compress_to_fit(raw, max_bytes=2048, preset="qr-code")
print(fit.quality_used, fit.reached_target)
```

## Accepted String Values

- Preset: `"qr-code"`, `"qr-code-match"`, `"print"`, `"display"`
- Crop mode: `"heuristic"`, `"none"`, `"face-detection"`
- Output format: `"webp"`, `"jpeg"`

## Exception Hierarchy

All library errors inherit from `IdPhotoError`, which inherits from `Exception`:

| Exception | Raised When |
| --- | --- |
| `DecodeError` | Input cannot be decoded as an image |
| `UnsupportedFormatError` | Input format is not JPEG, PNG, or WebP |
| `ZeroDimensionsError` | Image dimensions are zero |
| `EncodeError` | Output image cannot be encoded |
| `InvalidQualityError` | Quality is outside 0.0–1.0 |
| `InvalidMaxDimensionError` | max_dimension is 0 |

Argument validation errors (invalid preset, crop_mode, or output_format strings) raise `ValueError`.

```python
try:
    idphoto.compress(b"not an image")
except idphoto.DecodeError as e:
    print(e)  # caught as specific type
except idphoto.IdPhotoError:
    print("some other library error")
```

## Type Stubs

The package ships with a `py.typed` marker and `.pyi` stubs for full type checker support.
