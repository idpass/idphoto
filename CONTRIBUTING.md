# Contributing to idphoto

Thanks for contributing to `idphoto`.

## Development Setup

### Prerequisites

- Rust 1.75+
- Git LFS
- Python 3.9+ (for Python binding tests)

```sh
git lfs install
git clone https://github.com/idpass/idphoto.git
cd idphoto
```

## Build Instructions

### Core library

```sh
# Core crate only
cargo build -p idphoto

# Core crate with bundled SeetaFace backend
cargo build -p idphoto --features rustface
```

### Python bindings

```sh
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

### WASM bindings

```sh
rustup target add wasm32-unknown-unknown
cargo build -p idphoto-wasm --target wasm32-unknown-unknown
```

### JNI bindings

```sh
cargo build -p idphoto-jni
```

## Test Instructions

```sh
# Rust formatting + linting
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Rust test suite
cargo test --workspace --all-features

# Python binding tests
pip install maturin pytest
maturin develop --manifest-path bindings/python/Cargo.toml
pytest bindings/python/tests/ -v

# WASM compile check
cargo check -p idphoto-wasm --target wasm32-unknown-unknown

# WASM binding tests (requires wasm-bindgen test runner and a wasm C toolchain)
cargo install wasm-bindgen-cli --version 0.2.108 --locked
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p idphoto-wasm --target wasm32-unknown-unknown
```

## Code Style

- Use standard `rustfmt` formatting.
- Keep `clippy` warnings at zero (`-D warnings`).
- Add tests with behavior changes.
- Prefer clear, explicit errors over silent fallbacks.

## Pull Request Process

- Keep PRs focused and scoped.
- Update docs when behavior, APIs, or defaults change.
- Ensure CI passes before requesting review.
- Use descriptive commit messages and PR titles.
- Link related issues with `Fixes #<id>` when applicable.
