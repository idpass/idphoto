# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reusable LFW model comparison script (`bindings/python/examples/lfw_model_compare.py`) for int8, official InsightFace, and optional OpenCV SFace evaluation.
- Extended ArcFace QR evaluation script (`bindings/python/examples/arcface_qr_eval.py`) and testing docs for byte-budget quality analysis.
- Model asset download and licensing guidance (`model/README.md`), with docs-site navigation entry.

### Changed

- WebP encoding is now lossy and honors the configured quality value.
- SDK documentation updated for current bindings APIs (Python object attributes, WASM options object usage, and package naming).
- Python SDK now exposes idiomatic enums and typed options (`Preset`, `CropMode`, `OutputFormat`, `CompressOptions`, `CompressToFitOptions`) plus an object-style `IdPhoto` entrypoint while keeping function compatibility.
- WASM SDK now exposes an idiomatic client API (`createIdPhoto` + `IdPhoto` class) and exported constants for option values, while preserving `init`/`compress`/`compressToFit` compatibility exports.
- JVM SDK now provides separate language wrappers: Kotlin DSL wrapper (`IdPhoto`) and Java static wrapper (`JavaIdPhoto`) with Java-specific option/result types.
- Documentation site navigation is reorganized around Start/SDK/Core/Operations sections with a new SDK overview page, clearer cross-links, and language-first naming.
- Contributing/testing docs updated for WASM test prerequisites and runner usage.
- Error documentation clarified to match current decoder behavior for unsupported inputs.

## [0.1.0] - 2026-02-14

### Added

- Core Rust `idphoto` library for crop, resize, and compression of identity photos.
- Face-aware crop mode with pluggable `FaceDetector` trait.
- Built-in SeetaFace backend (`rustface` feature) with bundled model.
- Presets for `QrCode`, `QrCodeMatch`, `Print`, and `Display` use cases.
- Byte-budget compression API (`compress_to_fit`) using binary search over quality.
- Python bindings (`idphoto` module) with smoke tests.
- WASM bindings (`idphoto-wasm`) with smoke tests.
- JNI/UniFFI bindings (`idphoto-jni`) with smoke tests.
- CI coverage for formatting, linting, tests, docs, WASM checks, MSRV, and dependency audit.
- Open-source governance docs: license, security policy, contributing guide, and code of conduct.

### Changed

- README clarified bundled model size and licensing details.
- Added third-party attribution for bundled SeetaFace model.
- Added fixture provenance notes for synthetic test images.
