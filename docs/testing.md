# Testing And Fixtures

## Rust Tests

```bash
cargo test --workspace --all-features
```

## Python Binding Tests

```bash
pip install maturin pytest
maturin develop --manifest-path bindings/python/Cargo.toml
pytest bindings/python/tests/ -v
```

## Fixture Strategy

`tests/fixtures/` contains synthetic portraits (AI-generated, non-real people) used to verify:

- Face detection behavior
- Crop positioning (including off-center variants)
- Output consistency for presets and formats

Reference outputs are stored under `tests/fixtures/output/` and can be regenerated from examples in `core/idphoto/examples/` when algorithm behavior changes intentionally.
