/**
 * Options for photo compression.
 *
 * All fields are optional. When a `preset` is specified its defaults apply,
 * and individual fields override them.
 */
export interface CompressOptions {
  /** Preset name: "qr-code", "qr-code-match", "print", or "display". */
  preset?: "qr-code" | "qr-code-match" | "print" | "display";
  /** Maximum output dimension in pixels (overrides preset). */
  maxDimension?: number;
  /** Compression quality between 0.0 and 1.0 (overrides preset). */
  quality?: number;
  /** Convert to grayscale (overrides preset). */
  grayscale?: boolean;
  /** Crop mode: "heuristic", "none", or "face-detection" (overrides preset). */
  cropMode?: "heuristic" | "none" | "face-detection";
  /** Output format: "webp" or "jpeg" (overrides preset). */
  format?: "webp" | "jpeg";
  /** Face detection crop margin multiplier (overrides preset). */
  faceMargin?: number;
}

/** Bounding box of a detected face in the original image. */
export interface FaceBounds {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly confidence: number;
}

/** Result of a compress() call. */
export interface CompressedPhoto {
  /** Compressed image bytes. */
  readonly data: Uint8Array;
  /** Output format used. */
  readonly format: "webp" | "jpeg";
  /** Output width in pixels. */
  readonly width: number;
  /** Output height in pixels. */
  readonly height: number;
  /** Size of the original input in bytes. */
  readonly originalSize: number;
  /** Face bounds if a face was detected, or null. */
  readonly faceBounds: FaceBounds | null;
}

/** Result of a compressToFit() call. */
export interface FitResult {
  /** The compressed photo. */
  readonly photo: CompressedPhoto;
  /** Quality value that was used to meet the byte budget. */
  readonly qualityUsed: number;
  /** Whether the output fits within the requested byte budget. */
  readonly reachedTarget: boolean;
}

/** Error codes that can be returned by idphoto functions. */
export type IdPhotoErrorCode =
  | "DECODE_ERROR"
  | "UNSUPPORTED_FORMAT"
  | "ZERO_DIMENSIONS"
  | "ENCODE_ERROR"
  | "INVALID_QUALITY"
  | "INVALID_MAX_DIMENSION"
  | "INVALID_OPTIONS";

/** An error thrown by idphoto functions, with a machine-readable code. */
export interface IdPhotoError extends Error {
  readonly code: IdPhotoErrorCode;
}

/**
 * Compress an identity photo.
 *
 * @param input - Raw image bytes (JPEG, PNG, or WebP).
 * @param options - Compression options (all fields optional).
 * @returns The compressed photo.
 */
export function compress(
  input: Uint8Array,
  options?: CompressOptions,
): CompressedPhoto;

/**
 * Compress an identity photo to fit within a byte budget.
 *
 * Uses binary search over quality to find the highest quality
 * that produces output within `maxBytes`.
 *
 * @param input - Raw image bytes (JPEG, PNG, or WebP).
 * @param maxBytes - Maximum output size in bytes.
 * @param options - Compression options (all fields optional).
 * @returns The fit result with the compressed photo and quality used.
 */
export function compressToFit(
  input: Uint8Array,
  maxBytes: number,
  options?: CompressOptions,
): FitResult;

/**
 * Initialize the WASM module.
 *
 * @param moduleOrPath - Optional path, URL, or pre-compiled WebAssembly.Module.
 * @returns A promise that resolves when initialization is complete.
 */
export default function init(
  moduleOrPath?: string | URL | Request | WebAssembly.Module,
): Promise<void>;
