# Plan: Rust Library Improvements from LFW Evaluation Findings

## Context

LFW evaluation revealed that face alignment and efficient face cropping are critical
for matching quality. The Rust library's compression pipeline has several opportunities
to improve performance and matching effectiveness, grounded in measured results.

---

## Phase 1: Cache `RustfaceDetector` model

Smallest, safest change with immediate value. Do this first to de-risk Phase 2.

### 1.1 Use `std::sync::LazyLock` for the parsed model

**File:** `core/idphoto/src/rustface_backend.rs`

**Problem:** `RustfaceDetector::new()` parses the ~1.2MB SeetaFace model bytes on
every call. The implicit internal call path in `detect_face` (`compress.rs:33`)
creates a new `RustfaceDetector` per image. External callers using
`PhotoCompressor::face_detector(Box::new(RustfaceDetector::new()))` pay the cost
once per compressor (which they control), but the built-in fallback path does not
benefit from that.

**Fix:**
- Add a `static LazyLock<rustface::Model>` that parses the model bytes once.
- Remove the `model: rustface::Model` field from `RustfaceDetector`. The struct
  becomes a zero-size marker type.
- In `detect()`, clone from the static `LazyLock` to create the per-call detector
  (rustface's API requires ownership of a `Model` per `create_detector_with_model`).
- `LazyLock` is stable since Rust 1.80. Bump workspace `rust-version` from `"1.75"`
  to `"1.80"` in the root `Cargo.toml`.

### 1.2 Verify no behavioral change

Run existing tests. Caching changes initialization timing but not detection results.

---

## Phase 2: Square crop for matching mode

### 2.1 Use 1:1 aspect ratio when `face_margin < 1.5`

**File:** `core/idphoto/src/compress.rs`, `compute_face_crop` function

**Problem:** Face recognition models take square input (112x112). The current
3:4 aspect ratio wastes ~25% of pixels on non-face area (shoulders/hair) in
matching mode. At 48px max dimension, 3:4 gives 36x48 = 1,728 pixels. Square
gives 48x48 = 2,304 pixels, a 33% increase in face information.

**Fix:** In `compute_face_crop`, replace the hardcoded `portrait_aspect` local
variable with a conditional:
```rust
let aspect = if face_margin < 1.5 { 1.0 } else { 3.0 / 4.0 };
```

This variable is used in **three places** within `compute_face_crop`:
1. Line 76: `desired_crop_w = desired_crop_h * aspect` (initial sizing)
2. Line 83: `if (crop_w as f64 / crop_h as f64) > aspect` (clamp check)
3. Lines 85, 88: correction branches that re-enforce the aspect after clamping

All three must use the conditional `aspect` value. If only line 76 is changed,
the clamping block on lines 83-91 will re-snap a 1:1 crop back to 3:4 on narrow
images.

Also update the comment at `resize_image` line 188 which says "For
`CropMode::Heuristic`, the aspect is 3:4 so height > width." After this change,
matching-mode crops produce square output where height == width.

### 2.2 Update tests for square crop

- Add a test that `compute_face_crop` with `face_margin=1.3` produces a ~1:1
  aspect ratio.
- Verify the existing 3:4 test (`compute_face_crop_produces_3_4_aspect`) still
  passes with `face_margin=2.0`.

---

## Phase 3: Split `compress_to_fit` into prep + encode

### 3.1 Extract a `PreparedImage` struct and preparation function

**File:** `core/idphoto/src/compress.rs`

**Problem:** `compress_to_fit` calls `compress_pipeline` up to 9 times (worst case:
8 binary search iterations + 1 fallback). Each call re-decodes the image, re-runs
face detection, re-crops, re-resizes, re-flattens, and re-applies grayscale. Only
the encode step depends on quality. In the common case it's fewer calls, but the
decode + detect overhead is real for every iteration.

**Fix:** Split `compress_pipeline` into two `pub(crate)` functions:
- `prepare_image(...)` -> `PreparedImage` (decode, crop, resize, flatten, grayscale)
- `encode_prepared(prepared, format, quality)` -> `CompressedPhoto`

Both the struct and both functions must be `pub(crate)` since `compress_to_fit` in
`lib.rs` calls them across the module boundary.

**`PreparedImage` struct:**
```rust
pub(crate) struct PreparedImage {
    pub(crate) rgb: RgbImage,
    pub(crate) grayscale: bool,
    pub(crate) original_size: usize,
    pub(crate) face_bounds: Option<FaceBounds>,
}
```

**Important:** `apply_grayscale` is called during preparation, so `rgb` already
contains grayscale pixel data (R=G=B) when `grayscale=true`. The `grayscale: bool`
field is carried forward solely for the JPEG encoding path in `encode_image`, which
uses single-channel Luma8 instead of Rgb8 triplets. It must NOT be used to re-apply
grayscale conversion.

The `original_size` field keeps `encode_prepared` self-contained so it can construct
a complete `CompressedPhoto` without external context.

The existing `compress_pipeline` function is preserved. It calls `prepare_image`
then `encode_prepared` internally. No change to its signature or behavior.

### 3.2 Update `compress_to_fit` in `lib.rs`

**File:** `core/idphoto/src/lib.rs`

Call `compress::prepare_image` once before the binary search loop, then call
`compress::encode_prepared` per iteration. The `self.input` is consumed by
`prepare_image`, so the fallback path (line 384) also uses `encode_prepared`
instead of a fresh `compress_pipeline` call.

---

## Phase 4: Add `CropMode::DetectOnly`

### 4.1 Add the variant

**Files:** `core/idphoto/src/lib.rs`, `core/idphoto/src/compress.rs`

**Problem:** `CropMode::None` skips face detection entirely. A caller may want the
full image preserved while still getting face bounds metadata (for downstream
alignment or verification UIs).

**Fix:** Add `CropMode::DetectOnly`:
```rust
pub enum CropMode {
    Heuristic,
    FaceDetection,
    /// Detect faces and report bounds, but do not crop the image.
    DetectOnly,
    None,
}
```

In `apply_crop`, `DetectOnly` differs from `FaceDetection` in two ways:
1. It does NOT crop the image regardless of whether a face is found.
2. It does NOT fall back to heuristic crop when no face is found.

When a face is found: face_bounds is populated, image is returned uncropped,
crop_offset is (0, 0). When no face is found: face_bounds is None, image is
returned uncropped. This is different from `FaceDetection` which falls back to
heuristic crop when no face is found.

### 4.2 Expose in Python bindings

**File:** `bindings/python/src/lib.rs`

Add `"detect-only"` to `string_to_crop_mode` (follows existing kebab-case convention:
`"face-detection"`, `"detect-only"`).

**File:** `bindings/python/idphoto/__init__.py`

Add `DETECT_ONLY = "detect-only"` to the `CropMode` enum.

### 4.3 Add Rust tests

- `CropMode::DetectOnly` with a mock detector that finds a face: returns
  face_bounds in output coordinates, image dimensions match original (no crop).
- `CropMode::DetectOnly` with an empty mock detector: returns None face_bounds,
  image dimensions match original (no heuristic fallback).

---

## Task checklist

### Phase 1: Cache RustfaceDetector
- [x] 1.1 Cache parsed model with LazyLock, remove model field from struct
- [x] 1.2 Bump workspace rust-version to 1.80, verify tests pass

### Phase 2: Square crop for matching
- [x] 2.1 Conditional aspect ratio in all three uses within `compute_face_crop`
- [x] 2.2 Update `resize_image` comment about square crops
- [x] 2.3 Add test for 1:1 aspect when face_margin=1.3, verify 3:4 test still passes

### Phase 3: Split compress_to_fit
- [x] 3.1 Extract `PreparedImage`, `prepare_image`, `encode_prepared` (all pub(crate))
- [x] 3.2 Update `compress_to_fit` to use prep-then-encode pattern

### Phase 4: CropMode::DetectOnly
- [x] 4.1 Add DetectOnly variant to CropMode and apply_crop (no fallback to heuristic)
- [x] 4.2 Expose "detect-only" in Python bindings (Rust + Python enum)
- [x] 4.3 Add Rust tests for DetectOnly with face found and no face found
