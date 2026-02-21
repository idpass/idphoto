/** Common preset values. */
export declare const Preset: {
  readonly QR_CODE: "qr-code";
  readonly QR_CODE_MATCH: "qr-code-match";
  readonly PRINT: "print";
  readonly DISPLAY: "display";
};

/** Preset name. */
export type Preset = (typeof Preset)[keyof typeof Preset];

/** Common crop mode values. */
export declare const CropMode: {
  readonly HEURISTIC: "heuristic";
  readonly NONE: "none";
  readonly FACE_DETECTION: "face-detection";
};

/** Crop mode name. */
export type CropMode = (typeof CropMode)[keyof typeof CropMode];

/** Output format values. */
export declare const OutputFormat: {
  readonly WEBP: "webp";
  readonly JPEG: "jpeg";
};

/** Output format name. */
export type OutputFormat = (typeof OutputFormat)[keyof typeof OutputFormat];

/** Error codes that can be returned by idphoto functions. */
export declare const IdPhotoErrorCode: {
  readonly DECODE_ERROR: "DECODE_ERROR";
  readonly UNSUPPORTED_FORMAT: "UNSUPPORTED_FORMAT";
  readonly ZERO_DIMENSIONS: "ZERO_DIMENSIONS";
  readonly ENCODE_ERROR: "ENCODE_ERROR";
  readonly INVALID_QUALITY: "INVALID_QUALITY";
  readonly INVALID_MAX_DIMENSION: "INVALID_MAX_DIMENSION";
  readonly INVALID_OPTIONS: "INVALID_OPTIONS";
};

/** Error code union. */
export type IdPhotoErrorCode =
  (typeof IdPhotoErrorCode)[keyof typeof IdPhotoErrorCode];

/** Options for photo compression. */
export interface CompressOptions {
  /** Preset name. */
  preset?: Preset;
  /** Maximum output dimension in pixels (overrides preset). */
  maxDimension?: number;
  /** Compression quality between 0.0 and 1.0 (overrides preset). */
  quality?: number;
  /** Convert to grayscale (overrides preset). */
  grayscale?: boolean;
  /** Crop mode (overrides preset). */
  cropMode?: CropMode;
  /** Output format (overrides preset). */
  format?: OutputFormat;
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
  readonly format: OutputFormat;
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
  readonly photo: CompressedPhoto;
  readonly qualityUsed: number;
  readonly reachedTarget: boolean;
}

/** An error thrown by idphoto functions, with a machine-readable code. */
export interface IdPhotoError extends Error {
  readonly code: IdPhotoErrorCode;
}

/**
 * Initialize the WASM module.
 *
 * Subsequent calls return the same promise.
 */
declare function init(
  moduleOrPath?: string | URL | Request | WebAssembly.Module,
): Promise<void>;

/** Compress an identity photo. Requires prior init. */
export function compress(
  input: Uint8Array,
  options?: CompressOptions,
): CompressedPhoto;

/** Compress an identity photo to fit within a byte budget. Requires prior init. */
export function compressToFit(
  input: Uint8Array,
  maxBytes: number,
  options?: CompressOptions,
): FitResult;

/** Client-style API for JS/TS users. */
export class IdPhoto {
  compress(input: Uint8Array, options?: CompressOptions): CompressedPhoto;
  compressToFit(
    input: Uint8Array,
    maxBytes: number,
    options?: CompressOptions,
  ): FitResult;
}

/** Initialize and return an IdPhoto client instance. */
export function createIdPhoto(
  moduleOrPath?: string | URL | Request | WebAssembly.Module,
): Promise<IdPhoto>;

export { init };
export default init;
