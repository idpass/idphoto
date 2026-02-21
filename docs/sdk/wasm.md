# TypeScript/JavaScript SDK

Package: `@idpass/idphoto-wasm`

Implementation detail: this SDK runs via WebAssembly (WASM) and is built from `bindings/wasm`.

See [SDK Overview](index.md) for language selection and API shape comparison.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build bindings/wasm --target bundler
```

## API

The package exposes a client-style API:

- `createIdPhoto()` initializes the runtime and returns an `IdPhoto` client
- `IdPhoto.compress(input, options?)`
- `IdPhoto.compressToFit(input, maxBytes, options?)`

```typescript
import { createIdPhoto, Preset } from "@idpass/idphoto-wasm";

const idphoto = await createIdPhoto();
const input = new Uint8Array(await file.arrayBuffer());

const result = idphoto.compress(input, { preset: Preset.QR_CODE });
const fit = idphoto.compressToFit(input, 2048, { preset: Preset.QR_CODE });
```

## Functional API (Compatible)

The original functional API is still exported for compatibility:

- `init()`
- `compress(input, options?)`
- `compressToFit(input, maxBytes, options?)`

```typescript
import init, { compress, compressToFit } from "@idpass/idphoto-wasm";

await init();
const result = compress(input, { preset: "qr-code" });
const fit = compressToFit(input, 2048, { preset: "qr-code" });
```

## Constants And Options

Use exported constants to avoid string literals:

- `Preset` (`QR_CODE`, `QR_CODE_MATCH`, `PRINT`, `DISPLAY`)
- `CropMode` (`HEURISTIC`, `NONE`, `FACE_DETECTION`)
- `OutputFormat` (`WEBP`, `JPEG`)

```typescript
interface CompressOptions {
  preset?: Preset;
  maxDimension?: number;
  quality?: number; // 0.0 – 1.0
  grayscale?: boolean;
  cropMode?: CropMode;
  format?: OutputFormat;
  faceMargin?: number;
}
```

## Return Types

`compress` returns:

- `data` (`Uint8Array`) — compressed image bytes
- `format` (`"webp" | "jpeg"`)
- `width`, `height` (`number`)
- `originalSize` (`number`)
- `faceBounds` (`{ x, y, width, height, confidence } | null`)

`compressToFit` returns:

- `photo` (`CompressedPhoto`)
- `qualityUsed` (`number`)
- `reachedTarget` (`boolean`)

## Error Handling

Errors are standard `Error` objects with a machine-readable `code`:

| Code | Meaning |
| --- | --- |
| `DECODE_ERROR` | Input cannot be decoded |
| `UNSUPPORTED_FORMAT` | Reserved for unsupported-format mapping from core |
| `ZERO_DIMENSIONS` | Image dimensions are zero |
| `ENCODE_ERROR` | Output encoding failed |
| `INVALID_QUALITY` | Quality outside 0.0–1.0 |
| `INVALID_MAX_DIMENSION` | maxDimension is 0 |
| `INVALID_OPTIONS` | Options object is malformed |

Note: unsupported input formats are currently reported as `DECODE_ERROR` by the core decoder path.

```typescript
import { IdPhotoErrorCode, createIdPhoto } from "@idpass/idphoto-wasm";

const idphoto = await createIdPhoto();

try {
  idphoto.compress(input, { preset: "qr-code" });
} catch (e) {
  if ((e as { code?: string }).code === IdPhotoErrorCode.INVALID_OPTIONS) {
    console.error("invalid options", e);
  }
}
```
