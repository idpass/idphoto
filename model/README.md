# Model Assets

This directory contains face-related model files used by `idphoto`.

Large model binaries are typically downloaded locally and not committed.

## Files

- `seeta_fd_frontal_v1.0.bin` (bundled via Git LFS)
- `arcfaceresnet100-11-int8.onnx` (optional, downloaded for LFW/model comparisons)
- `face_recognition_sface_2021dec.onnx` (optional, downloaded for OpenCV FaceRecognizerSF comparisons)

## How To Get Them

### 1) Bundled Seeta model (for Rust face detection)

If missing after clone, pull via LFS:

```bash
git lfs pull --include="model/seeta_fd_frontal_v1.0.bin"
```

### 2) ONNXModelZoo ArcFace int8 model

```bash
curl -L \
  https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx \
  -o model/arcfaceresnet100-11-int8.onnx
```

Or let the script download it:

```bash
python bindings/python/examples/lfw_model_compare.py \
  --download-int8-model \
  --int8-model-path model/arcfaceresnet100-11-int8.onnx
```

### 3) OpenCV SFace recognition model

```bash
curl -L \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  -o model/face_recognition_sface_2021dec.onnx
```

Or let the script download it:

```bash
python bindings/python/examples/lfw_model_compare.py \
  --include-opencv \
  --download-opencv-model \
  --opencv-model-path model/face_recognition_sface_2021dec.onnx
```

### 4) InsightFace `w600k_r50` (used by `buffalo_l`)

Recommended: let InsightFace download via its own model management:

```bash
python - <<'PY'
import insightface
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Downloaded to ~/.insightface/models/buffalo_l")
PY
```

`lfw_model_compare.py` defaults to:

- `~/.insightface/models/buffalo_l/w600k_r50.onnx`

## License Notes

- `seeta_fd_frontal_v1.0.bin`: BSD 2-Clause (SeetaFace).  
  See `THIRD-PARTY-NOTICES`.

- `arcfaceresnet100-11-int8.onnx`: Apache-2.0 (as published on the Hugging Face model card).

- `face_recognition_sface_2021dec.onnx`: Apache-2.0 (see `models/face_recognition_sface/LICENSE` in `opencv/opencv_zoo`).

- InsightFace pretrained packs (including `buffalo_l` / `w600k_r50`):  
  InsightFace repository states that pretrained models are for non-commercial research purposes only; see the `License` section in `deepinsight/insightface` README. For commercial licensing, follow the contact addresses listed there.

Always verify upstream license terms before production or commercial use.
