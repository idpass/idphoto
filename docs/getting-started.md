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

## Pick Your SDK First

If you are unsure where to start, use [SDK Overview](sdk/index.md).

| Goal | Recommended SDK |
| --- | --- |
| Rust service/library integration | [Rust SDK](sdk/rust.md) |
| Python scripts, notebooks, ML pipelines | [Python SDK](sdk/python.md) |
| Browser or Node.js app integration | [TypeScript/JavaScript SDK](sdk/wasm.md) |
| Kotlin Android/JVM integration | [Kotlin SDK](sdk/kotlin.md) |
| Java Android/JVM integration | [Java SDK](sdk/java.md) |

## Build The Core Library

```bash
# Core (no built-in face detector)
cargo build -p idphoto

# Core with bundled SeetaFace detector
cargo build -p idphoto --features rustface
```

## Build The Binding You Need

### Python

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

### TypeScript/JavaScript

Build target: WebAssembly (WASM).

```bash
rustup target add wasm32-unknown-unknown
cargo build -p idphoto-wasm --target wasm32-unknown-unknown --release
```

### Kotlin

Binding layer: UniFFI/JNI.

```bash
cargo build -p idphoto-jni
```

### Java

Binding layer: UniFFI/JNI.

```bash
cargo build -p idphoto-jni
```

## Verify With Tests

```bash
# Rust test suite
cargo test --workspace --all-features

# Python bindings tests
pip install maturin pytest
maturin develop --manifest-path bindings/python/Cargo.toml
pytest bindings/python/tests/ -v
```

## Next Steps

- Read [Presets and Byte Budgets](core/presets.md)
- Understand [Processing Pipeline](core/pipeline.md)
- Integrate using your [SDK page](sdk/index.md)

## Run Docs Locally

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs serve
```

Site runs at `http://127.0.0.1:8001`.
