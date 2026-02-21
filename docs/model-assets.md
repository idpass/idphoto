# Model Assets

This page summarizes model files used by `idphoto`.

Source of truth: [`model/README.md`](https://github.com/idpass/idphoto/blob/main/model/README.md).

## Included/Optional Files

- `model/seeta_fd_frontal_v1.0.bin` (bundled via Git LFS)
- `model/arcfaceresnet100-11-int8.onnx` (optional, evaluation tooling)
- `model/face_recognition_sface_2021dec.onnx` (optional, OpenCV evaluation tooling)

## Download Commands

Bundled Seeta model (if missing after clone):

```bash
git lfs pull --include="model/seeta_fd_frontal_v1.0.bin"
```

ArcFace int8 ONNX model:

```bash
curl -L \
  https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx \
  -o model/arcfaceresnet100-11-int8.onnx
```

OpenCV SFace model:

```bash
curl -L \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  -o model/face_recognition_sface_2021dec.onnx
```

## Licensing

- `seeta_fd_frontal_v1.0.bin`: BSD 2-Clause (SeetaFace)
- `arcfaceresnet100-11-int8.onnx`: Apache-2.0 (upstream model card)
- `face_recognition_sface_2021dec.onnx`: Apache-2.0 (opencv_zoo)

Always verify upstream model licenses before production/commercial use.
