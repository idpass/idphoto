# Getting Started

## Prerequisites

- Rust 1.75+
- Git LFS (for bundled model and fixture files)
- Python 3.9+ (for Python bindings and docs tooling)

```bash
git lfs install
git clone https://github.com/idpass/idphoto.git
cd idphoto
```

## Build The Core Library

```bash
# Core (no built-in face detector)
cargo build -p idphoto

# Core with bundled SeetaFace detector
cargo build -p idphoto --features rustface
```

## Build Each Binding

### Python

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

### JavaScript / WASM

```bash
rustup target add wasm32-unknown-unknown
cargo build -p idphoto-wasm --target wasm32-unknown-unknown --release
```

### Kotlin / Java (JNI)

```bash
cargo build -p idphoto-jni
```

## Run Tests

```bash
# Rust test suite
cargo test --workspace --all-features

# Python bindings tests
pip install maturin pytest
maturin develop --manifest-path bindings/python/Cargo.toml
pytest bindings/python/tests/ -v
```

## Run Docs Locally

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs serve
```

Site runs at `http://127.0.0.1:8001`.
