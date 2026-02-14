# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
