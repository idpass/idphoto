#!/usr/bin/env python3
"""Evaluate QR-compressed idphoto outputs with ArcFace ONNX embeddings."""

from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import idphoto
except ImportError as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(
        "Could not import 'idphoto'. Build/install bindings first:\n"
        "  maturin develop --manifest-path bindings/python/Cargo.toml"
    ) from exc

import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/"
    "resolve/main/arcfaceresnet100-11-int8.onnx"
)
DEFAULT_MODEL_PATH = Path("model/arcfaceresnet100-11-int8.onnx")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ARC_FACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:
    """Face aligner for ArcFace-style 5-point landmark alignment."""

    def __init__(
        self,
        backend: str,
        providers: Sequence[str],
        input_size: int,
        det_size: int,
        fail_policy: str,
    ) -> None:
        self.backend = backend
        self.providers = list(providers)
        self.input_size = input_size
        self.det_size = det_size
        self.fail_policy = fail_policy
        self._detector = None

        if backend == "insightface":
            self._detector = self._load_insightface_detector()

    def _load_insightface_detector(self):
        try:
            import insightface  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime environment check
            raise SystemExit(
                "align-backend=insightface requires the 'insightface' package.\n"
                "Install it with:\n"
                "  pip install insightface opencv-python"
            ) from exc

        # Use CPU when CUDA provider is not requested.
        ctx_id = 0 if any(p.startswith("CUDA") for p in self.providers) else -1
        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=self.providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        return app

    def _template(self) -> np.ndarray:
        scale = self.input_size / 112.0
        return ARC_FACE_TEMPLATE_112 * scale

    def align_to_rgb(self, image: Image.Image) -> tuple[np.ndarray, str]:
        """Return aligned RGB image and alignment status."""
        if self.backend == "none":
            aligned = np.asarray(resize_and_pad_rgb(image, self.input_size), dtype=np.uint8)
            return aligned, "pad-resize"

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self._align_with_insightface(rgb)

    def _align_with_insightface(self, rgb: np.ndarray) -> tuple[np.ndarray, str]:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Tiny QR crops (e.g., 36x48) need aggressive temporary upscaling for
        # stable landmark detection.
        min_side = min(bgr.shape[0], bgr.shape[1])
        scale = max(1, int(math.ceil(224 / max(min_side, 1))))
        if scale > 1:
            detect_bgr = cv2.resize(
                bgr,
                dsize=(bgr.shape[1] * scale, bgr.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            detect_bgr = bgr

        faces = self._detector.get(detect_bgr)
        if not faces:
            return self._on_align_failure(rgb, reason="no face detected")

        def face_score(face) -> tuple:
            x1, y1, x2, y2 = face.bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            return float(face.det_score), area

        face = max(faces, key=face_score)
        landmarks = np.asarray(face.kps, dtype=np.float32)
        if scale > 1:
            landmarks = landmarks / float(scale)

        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self._template(),
            method=cv2.LMEDS,
        )
        if matrix is None:
            return self._on_align_failure(rgb, reason="affine estimation failed")

        aligned_bgr = cv2.warpAffine(
            bgr,
            matrix,
            (self.input_size, self.input_size),
            borderValue=0.0,
        )
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return aligned_rgb.astype(np.uint8), "landmarks"

    def _on_align_failure(self, rgb: np.ndarray, reason: str) -> tuple[np.ndarray, str]:
        if self.fail_policy == "error":
            raise RuntimeError(f"Face alignment failed ({reason})")
        pil_image = Image.fromarray(rgb, mode="RGB")
        fallback = resize_and_pad_rgb(pil_image, self.input_size)
        return np.asarray(fallback, dtype=np.uint8), f"fallback:{reason}"


@dataclass
class Variant:
    name: str
    data: bytes
    width: int
    height: int
    quality_used: Optional[float] = None
    reached_target: Optional[bool] = None


def parse_budgets(raw_value: str) -> List[int]:
    if not raw_value.strip():
        return []
    budgets: List[int] = []
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"Budget values must be positive integers, got: {value}"
            )
        budgets.append(value)
    return budgets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ArcFace cosine similarity between a baseline face image and "
            "QR-oriented idphoto outputs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image files and/or directories",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local ArcFace ONNX model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="Download URL used when model is missing",
    )
    parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download model automatically if missing (default: enabled)",
    )
    parser.add_argument(
        "--preset",
        default="qr-code-match",
        choices=["qr-code", "qr-code-match", "print", "display"],
        help="idphoto preset for QR variants (default: qr-code-match)",
    )
    parser.add_argument(
        "--max-bytes",
        default="2048,1536,1024",
        help=(
            "Comma-separated budgets for compress_to_fit variants "
            "(default: 2048,1536,1024). Use empty string to disable."
        ),
    )
    parser.add_argument(
        "--variant-output-format",
        default=None,
        choices=["jpeg", "webp"],
        help="Optional output format override for tested variants.",
    )
    parser.add_argument(
        "--variant-max-dimension",
        type=int,
        default=None,
        help="Optional max_dimension override for tested variants.",
    )
    parser.add_argument(
        "--variant-grayscale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional grayscale override for tested variants.",
    )
    parser.add_argument(
        "--variant-crop-mode",
        default=None,
        choices=["face-detection", "heuristic", "none"],
        help="Optional crop_mode override for tested variants.",
    )
    parser.add_argument(
        "--variant-face-margin",
        type=float,
        default=None,
        help="Optional face_margin override for tested variants.",
    )
    parser.add_argument(
        "--align-backend",
        default="insightface",
        choices=["insightface", "none"],
        help=(
            "Face alignment backend. 'insightface' does ArcFace-style 5-point "
            "alignment. 'none' uses pad+resize only."
        ),
    )
    parser.add_argument(
        "--align-det-size",
        type=int,
        default=640,
        help="Detector input size for insightface backend (default: 640)",
    )
    parser.add_argument(
        "--align-fail-policy",
        default="fallback",
        choices=["fallback", "error"],
        help=(
            "Behavior when landmarks cannot be detected: 'fallback' uses pad+resize, "
            "'error' stops execution."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["pipeline", "original"],
        default="pipeline",
        help=(
            "Baseline embedding source. 'pipeline' keeps crop logic similar to QR "
            "output. 'original' uses the untouched source image."
        ),
    )
    parser.add_argument(
        "--baseline-preset",
        default="qr-code-match",
        choices=["qr-code", "qr-code-match", "print", "display"],
        help="Preset used to build baseline when --baseline-mode pipeline",
    )
    parser.add_argument(
        "--baseline-dimension",
        type=int,
        default=112,
        help="max_dimension used for pipeline baseline (default: 112)",
    )
    parser.add_argument(
        "--baseline-format",
        default="jpeg",
        choices=["jpeg", "webp"],
        help="Output format used for pipeline baseline (default: jpeg)",
    )
    parser.add_argument(
        "--baseline-quality",
        type=float,
        default=0.95,
        help="JPEG quality used for pipeline baseline (default: 0.95)",
    )
    parser.add_argument(
        "--baseline-grayscale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use grayscale for pipeline baseline (default: disabled)",
    )
    parser.add_argument(
        "--baseline-crop-mode",
        default="face-detection",
        choices=["face-detection", "heuristic", "none"],
        help="Crop mode used for pipeline baseline (default: face-detection)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=112,
        help="ArcFace input width/height (default: 112)",
    )
    parser.add_argument(
        "--input-normalization",
        default="none",
        choices=["none", "minus1to1", "zero_to_one"],
        help=(
            "Normalization before ONNX inference. "
            "For the provided int8 model use 'none' because normalization is inside "
            "the model graph."
        ),
    )
    parser.add_argument(
        "--color-order",
        default="rgb",
        choices=["rgb", "bgr"],
        help="Channel order before ONNX inference (default: rgb)",
    )
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers (default: CPUExecutionProvider)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan directories for image files (default: enabled)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args(argv)


def ensure_model(model_path: Path, model_url: str, download_model: bool) -> None:
    if model_path.exists():
        return
    if not download_model:
        raise FileNotFoundError(
            f"Model not found at {model_path}. Re-run with --download-model."
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {model_path} ...", file=sys.stderr)
    with urllib.request.urlopen(model_url) as response, model_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def parse_providers(raw_value: str) -> List[str]:
    providers = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime provider is required")
    return providers


def discover_input_files(inputs: Sequence[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            continue
        if not path.is_dir():
            continue
        iterator: Iterable[Path]
        if recursive:
            iterator = path.rglob("*")
        else:
            iterator = path.glob("*")
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(candidate)
    unique_files = sorted(set(files))
    if not unique_files:
        raise FileNotFoundError("No supported image files were found in inputs")
    return unique_files


def decode_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def resize_and_pad_rgb(image: Image.Image, input_size: int) -> Image.Image:
    square = pad_to_square(image.convert("RGB"))
    return square.resize((input_size, input_size), Image.Resampling.BILINEAR)


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    x = (side - width) // 2
    y = (side - height) // 2
    square.paste(image, (x, y))
    return square


def image_to_model_tensor(
    aligned_rgb: np.ndarray,
    input_type: str,
    normalization: str,
    color_order: str,
) -> np.ndarray:
    array = np.asarray(aligned_rgb, dtype=np.float32)
    if color_order == "bgr":
        array = array[..., ::-1]

    if normalization == "minus1to1":
        array = (array - 127.5) / 128.0
    elif normalization == "zero_to_one":
        array = array / 255.0

    nchw = np.transpose(array, (2, 0, 1))[None, ...]

    if input_type == "tensor(float)":
        return nchw.astype(np.float32)
    if input_type == "tensor(uint8)":
        return np.clip(np.round(nchw), 0, 255).astype(np.uint8)
    if input_type == "tensor(int8)":
        return np.clip(np.round(nchw), -128, 127).astype(np.int8)
    raise ValueError(f"Unsupported model input type: {input_type}")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Zero-norm embedding encountered")
    return vector / norm


def embed_image(
    session: ort.InferenceSession,
    input_name: str,
    input_type: str,
    image: Image.Image,
    aligner: FaceAligner,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    aligned_rgb, align_status = aligner.align_to_rgb(image)
    tensor = image_to_model_tensor(
        aligned_rgb=aligned_rgb,
        input_type=input_type,
        normalization=args.input_normalization,
        color_order=args.color_order,
    )
    output = session.run(None, {input_name: tensor})[0]
    embedding = np.asarray(output, dtype=np.float32).reshape(-1)
    return l2_normalize(embedding), align_status


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def bytes_to_base64_length(byte_count: int) -> int:
    return ((byte_count + 2) // 3) * 4


def build_pipeline_baseline(raw: bytes, args: argparse.Namespace) -> bytes:
    result = idphoto.compress(
        raw,
        preset=args.baseline_preset,
        max_dimension=args.baseline_dimension,
        quality=args.baseline_quality,
        grayscale=args.baseline_grayscale,
        crop_mode=args.baseline_crop_mode,
        output_format=args.baseline_format,
    )
    return bytes(result.data)


def variant_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"preset": args.preset}
    if args.variant_output_format is not None:
        kwargs["output_format"] = args.variant_output_format
    if args.variant_max_dimension is not None:
        kwargs["max_dimension"] = args.variant_max_dimension
    if args.variant_grayscale is not None:
        kwargs["grayscale"] = args.variant_grayscale
    if args.variant_crop_mode is not None:
        kwargs["crop_mode"] = args.variant_crop_mode
    if args.variant_face_margin is not None:
        kwargs["face_margin"] = args.variant_face_margin
    return kwargs


def variant_label(args: argparse.Namespace) -> str:
    parts = [args.preset]
    if args.variant_output_format is not None:
        parts.append(args.variant_output_format)
    if args.variant_max_dimension is not None:
        parts.append(f"{args.variant_max_dimension}px")
    if args.variant_grayscale is not None:
        parts.append("gray" if args.variant_grayscale else "color")
    if args.variant_crop_mode is not None:
        parts.append(args.variant_crop_mode)
    if args.variant_face_margin is not None:
        parts.append(f"margin{args.variant_face_margin:g}")
    return ",".join(parts)


def build_variants(raw: bytes, args: argparse.Namespace, budgets: Sequence[int]) -> List[Variant]:
    variants: List[Variant] = []
    kwargs = variant_kwargs(args)
    label = variant_label(args)

    preset_result = idphoto.compress(raw, **kwargs)
    variants.append(
        Variant(
            name=f"preset:{label}",
            data=bytes(preset_result.data),
            width=preset_result.width,
            height=preset_result.height,
        )
    )

    for budget in budgets:
        fit = idphoto.compress_to_fit(raw, max_bytes=budget, **kwargs)
        variants.append(
            Variant(
                name=f"fit:{budget}:{label}",
                data=bytes(fit.data),
                width=fit.width,
                height=fit.height,
                quality_used=fit.quality_used,
                reached_target=fit.reached_target,
            )
        )

    return variants


def format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def print_table(rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    columns = [
        "image",
        "variant",
        "bytes",
        "base64_bytes",
        "size_px",
        "variant_align",
        "baseline_align",
        "reached_target",
        "quality_used",
        "cosine_to_baseline",
        "l2_to_baseline",
    ]
    widths = {
        column: max(len(column), *(len(row.get(column, "")) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        line = " | ".join(row.get(column, "").ljust(widths[column]) for column in columns)
        print(line)


def print_summary(summary_cosines: Dict[str, List[float]], summary_bytes: Dict[str, List[int]]) -> None:
    print("\nVariant summary:")
    for variant in sorted(summary_cosines):
        cosine_values = summary_cosines[variant]
        byte_values = summary_bytes[variant]
        avg_cosine = mean(cosine_values) if cosine_values else float("nan")
        avg_bytes = mean(byte_values) if byte_values else float("nan")
        print(
            f"  {variant}: "
            f"avg_cosine={avg_cosine:.6f}, "
            f"avg_bytes={avg_bytes:.1f}, "
            f"samples={len(cosine_values)}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(args.max_bytes)

    ensure_model(args.model_path, args.model_url, args.download_model)
    providers = parse_providers(args.providers)
    session = ort.InferenceSession(str(args.model_path), providers=providers)

    model_inputs = session.get_inputs()
    if len(model_inputs) != 1:
        raise RuntimeError(f"Expected exactly one model input, got {len(model_inputs)}")
    model_input = model_inputs[0]
    input_name = model_input.name
    input_type = model_input.type

    aligner = FaceAligner(
        backend=args.align_backend,
        providers=providers,
        input_size=args.input_size,
        det_size=args.align_det_size,
        fail_policy=args.align_fail_policy,
    )

    input_files = discover_input_files(args.inputs, args.recursive)
    print(f"Model: {args.model_path}")
    print(f"Input tensor: name={input_name}, type={input_type}, shape={model_input.shape}")
    print(
        "Alignment: "
        f"backend={args.align_backend}, "
        f"det_size={args.align_det_size}, "
        f"fail_policy={args.align_fail_policy}"
    )
    print(f"Found {len(input_files)} image(s)")

    rows: List[Dict[str, str]] = []
    summary_cosines: Dict[str, List[float]] = {}
    summary_bytes: Dict[str, List[int]] = {}

    for image_path in input_files:
        raw = image_path.read_bytes()
        if args.baseline_mode == "original":
            baseline_bytes = raw
            baseline_name = "baseline:original"
        else:
            baseline_bytes = build_pipeline_baseline(raw, args)
            baseline_name = (
                f"baseline:pipeline({args.baseline_preset},"
                f"{args.baseline_dimension}px,{args.baseline_format})"
            )

        baseline_image = decode_rgb(baseline_bytes)
        baseline_embedding, baseline_align_status = embed_image(
            session=session,
            input_name=input_name,
            input_type=input_type,
            image=baseline_image,
            aligner=aligner,
            args=args,
        )

        variants = build_variants(raw, args, budgets)
        for variant in variants:
            variant_image = decode_rgb(variant.data)
            variant_embedding, variant_align_status = embed_image(
                session=session,
                input_name=input_name,
                input_type=input_type,
                image=variant_image,
                aligner=aligner,
                args=args,
            )
            cosine = cosine_similarity(baseline_embedding, variant_embedding)
            distance = l2_distance(baseline_embedding, variant_embedding)
            byte_count = len(variant.data)

            row = {
                "image": str(image_path),
                "variant": variant.name,
                "bytes": str(byte_count),
                "base64_bytes": str(bytes_to_base64_length(byte_count)),
                "size_px": f"{variant.width}x{variant.height}",
                "variant_align": variant_align_status,
                "baseline_align": baseline_align_status,
                "reached_target": (
                    ""
                    if variant.reached_target is None
                    else ("true" if variant.reached_target else "false")
                ),
                "quality_used": format_float(variant.quality_used, digits=4),
                "cosine_to_baseline": format_float(cosine, digits=6),
                "l2_to_baseline": format_float(distance, digits=6),
                "baseline": baseline_name,
            }
            rows.append(row)

            summary_cosines.setdefault(variant.name, []).append(cosine)
            summary_bytes.setdefault(variant.name, []).append(byte_count)

    print_table(rows)
    print_summary(summary_cosines, summary_bytes)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image",
                    "baseline",
                    "variant",
                    "bytes",
                    "base64_bytes",
                    "size_px",
                    "variant_align",
                    "baseline_align",
                    "reached_target",
                    "quality_used",
                    "cosine_to_baseline",
                    "l2_to_baseline",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
