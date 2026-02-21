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

## ArcFace Quality Check For QR Photos

Use this when you want to compare the face embedding from the QR-target photo against a higher-quality baseline.

The example script is:

- `bindings/python/examples/arcface_qr_eval.py`

Install dependencies:

```bash
pip install onnxruntime numpy pillow insightface opencv-python scikit-learn
```

Build the Python bindings if needed:

```bash
maturin develop --manifest-path bindings/python/Cargo.toml
```

Run the evaluator (downloads your requested model automatically if missing):

```bash
python bindings/python/examples/arcface_qr_eval.py \
  tests/fixtures \
  --model-path model/arcfaceresnet100-11-int8.onnx \
  --model-url https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx \
  --align-backend insightface \
  --preset qr-code-match \
  --max-bytes 2048,1536,1024 \
  --output-csv /tmp/arcface_qr_eval.csv
```

For strict QR byte limits (example: `<=600 bytes`) with JPEG quality search:

```bash
python bindings/python/examples/arcface_qr_eval.py \
  tests/fixtures tests/fixtures/offcenter \
  --no-recursive \
  --baseline-mode original \
  --align-backend insightface \
  --preset qr-code-match \
  --variant-output-format jpeg \
  --max-bytes 600 \
  --output-csv /tmp/arcface_qr_eval_600.csv
```

Notes:

- For `arcfaceresnet100-11-int8`, keep `--input-normalization none`. This model already includes `(x - 127.5) / 128` in-graph.
- The default alignment backend is `insightface`, which runs 5-point landmark alignment before ArcFace embedding.
- The script reports per-image cosine similarity (`cosine_to_baseline`) and byte sizes for each QR variant.
- `base64_bytes` is included to estimate payload growth if your QR pipeline base64-encodes image bytes.

## Reusable LFW Model Comparison Script

Use this when you want a repeatable benchmark of:

- `onnxmodelzoo/arcfaceresnet100-11-int8`
- official InsightFace recognition model (`w600k_r50` from `buffalo_l`)

on the same protocol:

- query = compressed QR-oriented image
- gallery = original image
- metric = cosine similarity

Script:

- `bindings/python/examples/lfw_model_compare.py`

Example full run:

```bash
python bindings/python/examples/lfw_model_compare.py \
  --subset test \
  --int8-model-path /tmp/arcface_model/arcfaceresnet100-11-int8.onnx \
  --official-model-path ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  --preset qr-code-match \
  --max-bytes 600 \
  --output-format webp \
  --grayscale \
  --max-dimension 48 \
  --crop-mode none \
  --align-backend none \
  --output-json /tmp/lfw_compare.json \
  --output-csv /tmp/lfw_compare_scores.csv
```

To compare compressed query against compressed gallery (same processing on both sides), add:

```bash
  --gallery-source compressed
```

To also include OpenCV FaceRecognizerSF (`opencv_zoo` SFace model), add:

```bash
  --include-opencv \
  --opencv-model-path model/face_recognition_sface_2021dec.onnx
```

Fast smoke run:

```bash
python bindings/python/examples/lfw_model_compare.py \
  --subset test \
  --max-pairs 100 \
  --crop-mode none \
  --align-backend none
```
