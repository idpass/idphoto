import initWasm, {
  compress as wasmCompress,
  compressToFit as wasmCompressToFit,
} from "./pkg/idphoto_wasm.js";

export const Preset = Object.freeze({
  QR_CODE: "qr-code",
  QR_CODE_MATCH: "qr-code-match",
  PRINT: "print",
  DISPLAY: "display",
});

export const CropMode = Object.freeze({
  HEURISTIC: "heuristic",
  NONE: "none",
  FACE_DETECTION: "face-detection",
});

export const OutputFormat = Object.freeze({
  WEBP: "webp",
  JPEG: "jpeg",
});

export const IdPhotoErrorCode = Object.freeze({
  DECODE_ERROR: "DECODE_ERROR",
  UNSUPPORTED_FORMAT: "UNSUPPORTED_FORMAT",
  ZERO_DIMENSIONS: "ZERO_DIMENSIONS",
  ENCODE_ERROR: "ENCODE_ERROR",
  INVALID_QUALITY: "INVALID_QUALITY",
  INVALID_MAX_DIMENSION: "INVALID_MAX_DIMENSION",
  INVALID_OPTIONS: "INVALID_OPTIONS",
});

let initPromise;
let initialized = false;

/**
 * Initialize the WASM module.
 *
 * Subsequent calls return the same promise.
 */
export function init(moduleOrPath) {
  if (!initPromise) {
    initPromise = Promise.resolve(initWasm(moduleOrPath)).then(() => {
      initialized = true;
    });
  }
  return initPromise;
}

function assertInitialized() {
  if (initialized) {
    return;
  }

  throw new Error(
    "@idpass/idphoto-wasm is not initialized. Call init() or await createIdPhoto() first.",
  );
}

/**
 * Compress an identity photo.
 *
 * Requires prior `await init()` or `await createIdPhoto()`.
 */
export function compress(input, options) {
  assertInitialized();
  return wasmCompress(input, options);
}

/**
 * Compress an identity photo to fit a byte budget.
 *
 * Requires prior `await init()` or `await createIdPhoto()`.
 */
export function compressToFit(input, maxBytes, options) {
  assertInitialized();
  return wasmCompressToFit(input, maxBytes, options);
}

/**
 * Idiomatic client-style API for JavaScript/TypeScript callers.
 */
export class IdPhoto {
  compress(input, options) {
    return compress(input, options);
  }

  compressToFit(input, maxBytes, options) {
    return compressToFit(input, maxBytes, options);
  }
}

/**
 * Initialize and return an `IdPhoto` client instance.
 */
export async function createIdPhoto(moduleOrPath) {
  await init(moduleOrPath);
  return new IdPhoto();
}

export default init;
