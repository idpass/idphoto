# Contributing

Primary contributor guidance is maintained in:

- [`CONTRIBUTING.md`](https://github.com/idpass/idphoto/blob/main/CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](https://github.com/idpass/idphoto/blob/main/CODE_OF_CONDUCT.md)
- [`SECURITY.md`](https://github.com/idpass/idphoto/blob/main/SECURITY.md)

## Docs Contributions

When changing docs:

1. Update pages under `docs/`
2. Run a strict local build
3. Ensure command examples reflect current behavior

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs build --strict
```
