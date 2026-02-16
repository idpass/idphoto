# JavaScript SDK (WASM)

Package target: `idphoto-wasm` (`bindings/wasm`)

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build bindings/wasm --target bundler
```

## API Surface

### `compress(input, options?)`

Compress an identity photo.

- `input` — `Uint8Array` of raw image bytes (JPEG, PNG, or WebP)
- `options` — optional `CompressOptions` object

Returns a `CompressedPhoto` plain object.

### `compressToFit(input, maxBytes, options?)`

Compress to fit within a byte budget. Uses binary search over quality.

- `input` — `Uint8Array` of raw image bytes
- `maxBytes` — maximum output size in bytes
- `options` — optional `CompressOptions` object

Returns a `FitResult` plain object.

## Options Object

All fields are optional. When a `preset` is specified its defaults apply, and individual fields override them.

```typescript
interface CompressOptions {
  preset?: "qr-code" | "qr-code-match" | "print" | "display";
  maxDimension?: number;
  quality?: number;       // 0.0 – 1.0
  grayscale?: boolean;
  cropMode?: "heuristic" | "none" | "face-detection";
  format?: "webp" | "jpeg";
  faceMargin?: number;
}
```

## Return Types

`compress` returns a plain JS object:

- `data` (`Uint8Array`) — compressed image bytes
- `format` (`"webp"` or `"jpeg"`)
- `width`, `height` (`number`)
- `originalSize` (`number`)
- `faceBounds` (`{ x, y, width, height, confidence }` or `null`)

`compressToFit` returns a plain JS object:

- `photo` — a `CompressedPhoto` object (same shape as above)
- `qualityUsed` (`number`)
- `reachedTarget` (`boolean`)

## Error Handling

Errors are thrown as standard `Error` objects with a machine-readable `code` property:

| Code | Meaning |
| --- | --- |
| `DECODE_ERROR` | Input cannot be decoded |
| `UNSUPPORTED_FORMAT` | Input is not JPEG, PNG, or WebP |
| `ZERO_DIMENSIONS` | Image dimensions are zero |
| `ENCODE_ERROR` | Output encoding failed |
| `INVALID_QUALITY` | Quality outside 0.0–1.0 |
| `INVALID_MAX_DIMENSION` | maxDimension is 0 |
| `INVALID_OPTIONS` | Options object is malformed |

```javascript
try {
  const result = compress(input, { preset: "qr-code" });
} catch (e) {
  console.error(e.code, e.message);
}
```

## Example

```javascript
import init, { compress, compressToFit } from "idphoto-wasm";

await init();

const input = new Uint8Array(await file.arrayBuffer());

const result = compress(input, { preset: "qr-code" });
console.log(result.width, result.height, result.data.length);

const fit = compressToFit(input, 2048, { preset: "qr-code" });
console.log(fit.qualityUsed, fit.reachedTarget);
```

## TypeScript

The package includes hand-written `.d.ts` declarations with full type information for `CompressOptions`, `CompressedPhoto`, `FaceBounds`, `FitResult`, and error codes.
