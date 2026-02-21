#!/usr/bin/env python3
"""Compare QR-compressed verification performance across ArcFace models on LFW."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from sklearn.datasets import fetch_lfw_pairs

from arcface_qr_eval import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_URL,
    FaceAligner,
    decode_rgb,
    ensure_model,
    image_to_model_tensor,
    l2_normalize,
    parse_providers,
)

DEFAULT_OFFICIAL_MODEL_PATH = (
    Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
)
DEFAULT_OPENCV_MODEL_PATH = Path("model/face_recognition_sface_2021dec.onnx")
DEFAULT_OPENCV_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
DEFAULT_OPERATING_POINTS = "0.10,0.15,0.20,0.25,0.30"
DEFAULT_FAR_CAPS = "0.10,0.05,0.01,0.001"


def parse_float_list(raw_value: str) -> List[float]:
    if not raw_value.strip():
        return []
    values: List[float] = []
    for token in raw_value.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two face-embedding models on LFW using compressed QR photos as "
            "query images and original photos as gallery images."
        )
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "test", "10_folds"],
        help="LFW subset passed to sklearn fetch_lfw_pairs (default: test).",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="LFW resize passed to fetch_lfw_pairs (default: 1.0).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Optional balanced pair limit for fast runs. Uses roughly half positive "
            "and half negative pairs."
        ),
    )
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=112,
        help="Input size for both models (default: 112).",
    )
    parser.add_argument(
        "--align-backend",
        default="none",
        choices=["none", "insightface"],
        help=(
            "Pre-embedding alignment backend. 'none' uses pad+resize and is much "
            "faster for LFW. 'insightface' uses landmark alignment."
        ),
    )
    parser.add_argument(
        "--align-det-size",
        type=int,
        default=640,
        help="Detector size for align-backend insightface (default: 640).",
    )
    parser.add_argument(
        "--align-fail-policy",
        default="fallback",
        choices=["fallback", "error"],
        help="Alignment failure behavior (default: fallback).",
    )
    parser.add_argument(
        "--input-normalization",
        default="none",
        choices=["none", "minus1to1", "zero_to_one"],
        help=(
            "Normalization for the ONNXModelZoo int8 model input. Keep 'none' for "
            "arcfaceresnet100-11-int8."
        ),
    )
    parser.add_argument(
        "--color-order",
        default="rgb",
        choices=["rgb", "bgr"],
        help="Color order before ONNXModelZoo inference (default: rgb).",
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to ONNXModelZoo model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--int8-model-url",
        default=DEFAULT_MODEL_URL,
        help="Download URL for ONNXModelZoo model if missing.",
    )
    parser.add_argument(
        "--download-int8-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download ONNXModelZoo model automatically if missing.",
    )
    parser.add_argument(
        "--official-model-path",
        type=Path,
        default=DEFAULT_OFFICIAL_MODEL_PATH,
        help=f"Path to official InsightFace recognition model (default: {DEFAULT_OFFICIAL_MODEL_PATH}).",
    )
    parser.add_argument(
        "--download-official-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If official model is missing, initialize FaceAnalysis to download model "
            "pack (default: enabled)."
        ),
    )
    parser.add_argument(
        "--official-pack",
        default="buffalo_l",
        help="InsightFace pack to initialize when downloading official model.",
    )
    parser.add_argument(
        "--include-opencv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate OpenCV FaceRecognizerSF (SFace model).",
    )
    parser.add_argument(
        "--opencv-model-path",
        type=Path,
        default=DEFAULT_OPENCV_MODEL_PATH,
        help=f"Path to OpenCV SFace model (default: {DEFAULT_OPENCV_MODEL_PATH}).",
    )
    parser.add_argument(
        "--opencv-model-url",
        default=DEFAULT_OPENCV_MODEL_URL,
        help="Download URL for OpenCV SFace model if missing.",
    )
    parser.add_argument(
        "--download-opencv-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download OpenCV SFace model automatically if missing.",
    )
    parser.add_argument(
        "--preset",
        default="qr-code-match",
        choices=["qr-code", "qr-code-match", "print", "display"],
        help="Compression preset for query images.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=600,
        help="Maximum compressed byte size for query images (default: 600).",
    )
    parser.add_argument(
        "--output-format",
        default="webp",
        choices=["jpeg", "webp"],
        help="Compressed output format (default: webp).",
    )
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use grayscale compressed query images (default: enabled).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=48,
        help="max_dimension for compressed query images (default: 48).",
    )
    parser.add_argument(
        "--crop-mode",
        default="none",
        choices=["face-detection", "heuristic", "none"],
        help="Compression crop mode for query images (default: none).",
    )
    parser.add_argument(
        "--gallery-source",
        default="original",
        choices=["original", "compressed"],
        help=(
            "Gallery source for scoring. 'original' = compressed vs original "
            "(verification style). 'compressed' = compressed vs compressed."
        ),
    )
    parser.add_argument(
        "--operating-points",
        default=DEFAULT_OPERATING_POINTS,
        help="Comma-separated thresholds to report (default: 0.10,0.15,0.20,0.25,0.30).",
    )
    parser.add_argument(
        "--far-caps",
        default=DEFAULT_FAR_CAPS,
        help="Comma-separated FAR caps for best TAR lookup (default: 0.10,0.05,0.01,0.001).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write per-model score rows.",
    )
    return parser.parse_args(argv)


def ensure_official_model(
    model_path: Path,
    download: bool,
    pack_name: str,
    providers: Sequence[str],
) -> None:
    if model_path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Official model not found at {model_path}. Re-run with "
            "--download-official-model."
        )

    try:
        import insightface  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment check
        raise SystemExit(
            "Official model download requires insightface package.\n"
            "Install it with:\n"
            "  pip install insightface opencv-python"
        ) from exc

    ctx_id = 0 if any(p.startswith("CUDA") for p in providers) else -1
    app = insightface.app.FaceAnalysis(name=pack_name, providers=list(providers))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected model at {model_path} after download, but file is still missing."
        )


def image_to_png_bytes(image_array: np.ndarray) -> bytes:
    array = np.asarray(image_array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    mode = "L" if array.ndim == 2 else "RGB"
    image = Image.fromarray(array, mode=mode)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def hash_key(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def prepare_pairs(
    subset: str,
    resize: float,
    max_pairs: Optional[int],
) -> tuple[List[Tuple[str, str, bool]], Dict[str, bytes]]:
    dataset = fetch_lfw_pairs(subset=subset, funneled=True, resize=resize, color=True)
    pairs = dataset.pairs
    labels = dataset.target.astype(bool)

    print(
        f"LFW loaded: subset={subset}, pair_count={len(pairs)}, shape={pairs.shape}, "
        f"dtype={pairs.dtype}",
        flush=True,
    )

    pair_entries: List[Tuple[str, str, bool]] = []
    positives: List[Tuple[str, str, bool]] = []
    negatives: List[Tuple[str, str, bool]] = []
    raw_by_key: Dict[str, bytes] = {}

    for index, (pair, is_same) in enumerate(zip(pairs, labels), start=1):
        raw_a = image_to_png_bytes(pair[0])
        raw_b = image_to_png_bytes(pair[1])
        key_a = hash_key(raw_a)
        key_b = hash_key(raw_b)
        raw_by_key.setdefault(key_a, raw_a)
        raw_by_key.setdefault(key_b, raw_b)

        entry = (key_a, key_b, bool(is_same))
        pair_entries.append(entry)
        if is_same:
            positives.append(entry)
        else:
            negatives.append(entry)

        if index % 200 == 0:
            print(
                f"Prepared {index}/{len(pairs)} pairs "
                f"(unique_images={len(raw_by_key)})",
                flush=True,
            )

    if max_pairs is not None and max_pairs > 0 and max_pairs < len(pair_entries):
        n_pos = max_pairs // 2
        n_neg = max_pairs - n_pos
        selected = positives[:n_pos] + negatives[:n_neg]
        pair_entries = selected
        print(
            f"Using balanced subset: requested={max_pairs}, "
            f"selected={len(pair_entries)} (pos={n_pos}, neg={n_neg})",
            flush=True,
        )

    used_keys = {key for pair in pair_entries for key in pair[:2]}
    trimmed_raw = {key: raw_by_key[key] for key in used_keys}
    print(
        f"Effective pair_count={len(pair_entries)}, unique_images={len(trimmed_raw)}",
        flush=True,
    )
    return pair_entries, trimmed_raw


def compress_unique_images(
    raw_by_key: Dict[str, bytes],
    args: argparse.Namespace,
) -> tuple[Dict[str, bytes], Dict[str, object]]:
    compressed_by_key: Dict[str, bytes] = {}
    sizes: List[int] = []
    qualities: List[float] = []
    reached = 0

    ordered_keys = sorted(raw_by_key)
    for index, key in enumerate(ordered_keys, start=1):
        raw = raw_by_key[key]
        fit = idphoto.compress_to_fit(
            raw,
            max_bytes=args.max_bytes,
            preset=args.preset,
            output_format=args.output_format,
            grayscale=args.grayscale,
            max_dimension=args.max_dimension,
            crop_mode=args.crop_mode,
        )
        compressed = bytes(fit.data)
        compressed_by_key[key] = compressed
        sizes.append(len(compressed))
        qualities.append(float(fit.quality_used))
        if bool(fit.reached_target):
            reached += 1

        if index % 200 == 0:
            print(f"Compressed {index}/{len(ordered_keys)} images", flush=True)

    stats = {
        "count": len(ordered_keys),
        "mean_bytes": mean(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "reached_target": reached,
        "mean_quality": mean(qualities),
        "min_quality": min(qualities),
        "max_quality": max(qualities),
    }
    return compressed_by_key, stats


def load_int8_model(
    model_path: Path,
    model_url: str,
    download_model: bool,
    providers: Sequence[str],
) -> tuple[ort.InferenceSession, str, str]:
    ensure_model(model_path, model_url, download_model)
    session = ort.InferenceSession(str(model_path), providers=list(providers))
    model_input = session.get_inputs()[0]
    return session, model_input.name, model_input.type


def load_official_model(
    model_path: Path,
    providers: Sequence[str],
):
    import insightface  # type: ignore

    model = insightface.model_zoo.get_model(str(model_path), providers=list(providers))
    ctx_id = 0 if any(p.startswith("CUDA") for p in providers) else -1
    model.prepare(ctx_id=ctx_id)
    return model


def ensure_opencv_model(
    model_path: Path,
    model_url: str,
    download_model: bool,
) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment check
        raise SystemExit(
            "OpenCV model evaluation requires opencv-python.\n"
            "Install it with:\n"
            "  pip install opencv-python"
        ) from exc

    if not hasattr(cv2, "FaceRecognizerSF_create"):
        raise SystemExit(
            "Your OpenCV build does not expose FaceRecognizerSF_create.\n"
            "Install/upgrade opencv-python (4.7+) and retry."
        )

    ensure_model(model_path, model_url, download_model)


def load_opencv_model(model_path: Path):
    import cv2  # type: ignore

    return cv2.FaceRecognizerSF_create(str(model_path), "")


def embed_opencv(model, aligned_rgb: np.ndarray, input_size: int) -> np.ndarray:
    import cv2  # type: ignore

    bgr = aligned_rgb[..., ::-1]
    if bgr.shape[0] != input_size or bgr.shape[1] != input_size:
        bgr = cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    feature = np.asarray(model.feature(bgr), dtype=np.float32).reshape(-1)
    return l2_normalize(feature)


def compute_metrics(
    positive_scores: List[float],
    negative_scores: List[float],
    far_caps: Sequence[float],
    operating_points: Sequence[float],
) -> Dict[str, object]:
    pos = np.array(positive_scores, dtype=np.float64)
    neg = np.array(negative_scores, dtype=np.float64)

    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Both positive and negative scores are required for metrics.")

    def rates(threshold: float) -> tuple[float, float, float]:
        far = float(np.mean(neg >= threshold))
        frr = float(np.mean(pos < threshold))
        tar = 1.0 - frr
        return tar, far, frr

    threshold_values = list(pos) + list(neg) + list(operating_points)
    threshold_values.append(float(neg.max()) + 1e-6)
    threshold_values.append(float(pos.min()) - 1e-6)
    thresholds = sorted(set(threshold_values))

    best_bal = None
    for threshold in thresholds:
        tar, far, frr = rates(threshold)
        balanced_accuracy = 1.0 - (far + frr) / 2.0
        if best_bal is None or balanced_accuracy > best_bal["balanced_accuracy"]:
            best_bal = {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy),
                "tar": float(tar),
                "far": float(far),
                "frr": float(frr),
            }

    eer = None
    for threshold in thresholds:
        tar, far, frr = rates(threshold)
        distance = abs(far - frr)
        if eer is None or distance < eer["distance"]:
            eer = {
                "threshold": float(threshold),
                "eer": float((far + frr) / 2.0),
                "far": float(far),
                "frr": float(frr),
                "distance": float(distance),
            }

    far_lookup: Dict[str, Optional[Dict[str, float]]] = {}
    for cap in far_caps:
        candidates: List[Dict[str, float]] = []
        for threshold in thresholds:
            tar, far, frr = rates(threshold)
            if far <= cap:
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "tar": float(tar),
                        "far": float(far),
                        "frr": float(frr),
                    }
                )
        if candidates:
            candidates.sort(key=lambda item: (item["tar"], item["threshold"]), reverse=True)
            far_lookup[f"{cap:g}"] = candidates[0]
        else:
            far_lookup[f"{cap:g}"] = None

    operating = []
    for threshold in operating_points:
        tar, far, frr = rates(threshold)
        operating.append(
            {
                "threshold": float(threshold),
                "tar": float(tar),
                "far": float(far),
                "frr": float(frr),
            }
        )

    return {
        "positive": {
            "count": int(len(pos)),
            "mean": float(pos.mean()),
            "min": float(pos.min()),
            "max": float(pos.max()),
        },
        "negative": {
            "count": int(len(neg)),
            "mean": float(neg.mean()),
            "min": float(neg.min()),
            "max": float(neg.max()),
        },
        "best_balanced": best_bal,
        "eer_approx": eer,
        "best_under_far": far_lookup,
        "operating_points": operating,
    }


def print_model_summary(name: str, metrics: Dict[str, object], far_caps: Sequence[float]) -> None:
    positive = metrics["positive"]
    negative = metrics["negative"]
    best_bal = metrics["best_balanced"]
    eer = metrics["eer_approx"]
    far_lookup = metrics["best_under_far"]

    print(f"[{name}]")
    print(
        "  positive cosine: "
        f"count={positive['count']} "
        f"mean={positive['mean']:.6f} "
        f"min={positive['min']:.6f} "
        f"max={positive['max']:.6f}"
    )
    print(
        "  negative cosine: "
        f"count={negative['count']} "
        f"mean={negative['mean']:.6f} "
        f"min={negative['min']:.6f} "
        f"max={negative['max']:.6f}"
    )
    print(
        "  best balanced: "
        f"thr={best_bal['threshold']:.6f} "
        f"bal_acc={best_bal['balanced_accuracy']:.6f} "
        f"TAR={best_bal['tar']:.6f} "
        f"FAR={best_bal['far']:.6f} "
        f"FRR={best_bal['frr']:.6f}"
    )
    print(
        "  eer approx: "
        f"thr={eer['threshold']:.6f} "
        f"EER={eer['eer']:.6f} "
        f"FAR={eer['far']:.6f} "
        f"FRR={eer['frr']:.6f}"
    )
    for cap in far_caps:
        item = far_lookup[f"{cap:g}"]
        if item is None:
            print(f"  FAR<={cap:g}: no threshold")
        else:
            print(
                f"  FAR<={cap:g}: "
                f"thr={item['threshold']:.6f} "
                f"TAR={item['tar']:.6f} "
                f"FAR={item['far']:.6f} "
                f"FRR={item['frr']:.6f}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    far_caps = parse_float_list(args.far_caps)
    operating_points = parse_float_list(args.operating_points)
    if not far_caps:
        raise argparse.ArgumentTypeError("At least one FAR cap is required.")
    if not operating_points:
        raise argparse.ArgumentTypeError("At least one operating point is required.")

    providers = parse_providers(args.providers)
    ensure_official_model(
        model_path=args.official_model_path,
        download=args.download_official_model,
        pack_name=args.official_pack,
        providers=providers,
    )
    if args.include_opencv:
        ensure_opencv_model(
            model_path=args.opencv_model_path,
            model_url=args.opencv_model_url,
            download_model=args.download_opencv_model,
        )

    pair_entries, raw_by_key = prepare_pairs(
        subset=args.subset,
        resize=args.resize,
        max_pairs=args.max_pairs,
    )
    compressed_by_key, compression_stats = compress_unique_images(raw_by_key, args)

    print(
        "Compression stats: "
        f"count={compression_stats['count']} "
        f"mean_bytes={compression_stats['mean_bytes']:.2f} "
        f"min_bytes={compression_stats['min_bytes']} "
        f"max_bytes={compression_stats['max_bytes']} "
        f"reached={compression_stats['reached_target']}/{compression_stats['count']} "
        f"mean_quality={compression_stats['mean_quality']:.4f}",
        flush=True,
    )

    print("Loading models...", flush=True)
    int8_session, int8_input_name, int8_input_type = load_int8_model(
        model_path=args.int8_model_path,
        model_url=args.int8_model_url,
        download_model=args.download_int8_model,
        providers=providers,
    )
    official_model = load_official_model(args.official_model_path, providers=providers)
    opencv_model = (
        load_opencv_model(args.opencv_model_path) if args.include_opencv else None
    )

    aligner = FaceAligner(
        backend=args.align_backend,
        providers=providers,
        input_size=args.input_size,
        det_size=args.align_det_size,
        fail_policy=args.align_fail_policy,
    )

    align_counts = {
        "orig": Counter(),
        "comp": Counter(),
    }

    embeddings: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        "int8": {"orig": {}, "comp": {}},
        "official": {"orig": {}, "comp": {}},
    }
    if opencv_model is not None:
        embeddings["opencv_sface"] = {"orig": {}, "comp": {}}

    ordered_keys = sorted(raw_by_key)
    print(f"Embedding {len(ordered_keys)} unique images for both models...", flush=True)
    for index, key in enumerate(ordered_keys, start=1):
        original_image = decode_rgb(raw_by_key[key])
        compressed_image = decode_rgb(compressed_by_key[key])

        original_rgb, original_align = aligner.align_to_rgb(original_image)
        compressed_rgb, compressed_align = aligner.align_to_rgb(compressed_image)
        align_counts["orig"][original_align] += 1
        align_counts["comp"][compressed_align] += 1

        original_tensor = image_to_model_tensor(
            aligned_rgb=original_rgb,
            input_type=int8_input_type,
            normalization=args.input_normalization,
            color_order=args.color_order,
        )
        compressed_tensor = image_to_model_tensor(
            aligned_rgb=compressed_rgb,
            input_type=int8_input_type,
            normalization=args.input_normalization,
            color_order=args.color_order,
        )
        original_int8 = int8_session.run(None, {int8_input_name: original_tensor})[0]
        compressed_int8 = int8_session.run(None, {int8_input_name: compressed_tensor})[0]
        embeddings["int8"]["orig"][key] = l2_normalize(
            np.asarray(original_int8, dtype=np.float32).reshape(-1)
        )
        embeddings["int8"]["comp"][key] = l2_normalize(
            np.asarray(compressed_int8, dtype=np.float32).reshape(-1)
        )

        original_bgr = original_rgb[..., ::-1]
        compressed_bgr = compressed_rgb[..., ::-1]
        original_official = np.asarray(official_model.get_feat(original_bgr), dtype=np.float32)
        compressed_official = np.asarray(
            official_model.get_feat(compressed_bgr), dtype=np.float32
        )
        embeddings["official"]["orig"][key] = l2_normalize(original_official.reshape(-1))
        embeddings["official"]["comp"][key] = l2_normalize(compressed_official.reshape(-1))

        if opencv_model is not None:
            embeddings["opencv_sface"]["orig"][key] = embed_opencv(
                opencv_model,
                original_rgb,
                args.input_size,
            )
            embeddings["opencv_sface"]["comp"][key] = embed_opencv(
                opencv_model,
                compressed_rgb,
                args.input_size,
            )

        if index % 200 == 0:
            print(f"Embedded {index}/{len(ordered_keys)} images", flush=True)

    results: Dict[str, Dict[str, object]] = {}
    score_rows: List[Dict[str, object]] = []

    gallery_embed_key = "orig" if args.gallery_source == "original" else "comp"

    model_names = ["int8", "official"]
    if opencv_model is not None:
        model_names.append("opencv_sface")

    for model_name in model_names:
        positives: List[float] = []
        negatives: List[float] = []

        for pair_index, (key_a, key_b, is_same) in enumerate(pair_entries, start=1):
            query_a = embeddings[model_name]["comp"][key_a]
            gallery_b = embeddings[model_name][gallery_embed_key][key_b]
            score_a_to_b = float(np.dot(query_a, gallery_b))

            query_b = embeddings[model_name]["comp"][key_b]
            gallery_a = embeddings[model_name][gallery_embed_key][key_a]
            score_b_to_a = float(np.dot(query_b, gallery_a))

            if is_same:
                positives.extend([score_a_to_b, score_b_to_a])
            else:
                negatives.extend([score_a_to_b, score_b_to_a])

            score_rows.append(
                {
                    "model": model_name,
                    "pair_index": pair_index,
                    "is_same": "true" if is_same else "false",
                    "score_a_to_b": score_a_to_b,
                    "score_b_to_a": score_b_to_a,
                }
            )

        results[model_name] = compute_metrics(
            positive_scores=positives,
            negative_scores=negatives,
            far_caps=far_caps,
            operating_points=operating_points,
        )

    print("\n=== Model Comparison ===")
    print(
        "Config: "
        f"subset={args.subset}, pair_count={len(pair_entries)}, "
        f"align_backend={args.align_backend}, "
        f"gallery_source={args.gallery_source}, "
        f"include_opencv={args.include_opencv}, "
        f"compression={args.preset}/{args.output_format}/gray={args.grayscale}/"
        f"max_dim={args.max_dimension}/max_bytes={args.max_bytes}/crop={args.crop_mode}"
    )
    print(f"Alignment counts (original): {dict(align_counts['orig'])}")
    print(f"Alignment counts (compressed): {dict(align_counts['comp'])}")

    labels = {
        "int8": "onnxmodelzoo-int8",
        "official": "insightface-official",
        "opencv_sface": "opencv-sface",
    }
    for model_name in model_names:
        print_model_summary(labels[model_name], results[model_name], far_caps)

    print("\nDelta vs int8:")
    int8_eer = results["int8"]["eer_approx"]["eer"]
    for model_name in model_names:
        if model_name == "int8":
            continue
        other_eer = results[model_name]["eer_approx"]["eer"]
        print(f"  {labels[model_name]} EER delta={other_eer - int8_eer:.6f}")
        for cap in far_caps:
            key = f"{cap:g}"
            int8_far = results["int8"]["best_under_far"][key]
            other_far = results[model_name]["best_under_far"][key]
            if int8_far is None or other_far is None:
                print(f"  {labels[model_name]} TAR@FAR<={cap:g} delta=nan")
            else:
                print(
                    f"  {labels[model_name]} TAR@FAR<={cap:g} delta="
                    f"{other_far['tar'] - int8_far['tar']:.6f}"
                )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "pair_index",
                    "is_same",
                    "score_a_to_b",
                    "score_b_to_a",
                ],
            )
            writer.writeheader()
            writer.writerows(score_rows)
        print(f"Wrote CSV: {args.output_csv}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "subset": args.subset,
                "resize": args.resize,
                "pair_count": len(pair_entries),
                "max_pairs": args.max_pairs,
                "providers": list(providers),
                "align_backend": args.align_backend,
                "align_det_size": args.align_det_size,
                "align_fail_policy": args.align_fail_policy,
                "gallery_source": args.gallery_source,
                "include_opencv": args.include_opencv,
                "opencv_model_path": str(args.opencv_model_path),
                "input_size": args.input_size,
                "compression": {
                    "preset": args.preset,
                    "max_bytes": args.max_bytes,
                    "output_format": args.output_format,
                    "grayscale": args.grayscale,
                    "max_dimension": args.max_dimension,
                    "crop_mode": args.crop_mode,
                },
                "far_caps": far_caps,
                "operating_points": operating_points,
            },
            "compression_stats": compression_stats,
            "alignment_counts": {
                "original": dict(align_counts["orig"]),
                "compressed": dict(align_counts["comp"]),
            },
            "models": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
