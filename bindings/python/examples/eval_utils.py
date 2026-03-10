"""Shared utilities for face-matching evaluation scripts."""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("Near-zero-norm embedding encountered")
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def parse_providers(raw_value: str) -> List[str]:
    providers = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime provider is required")
    return providers


def parse_budgets(raw_value: str) -> List[int]:
    import argparse

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


def decode_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


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


def resize_and_pad_rgb(image: Image.Image, input_size: int) -> Image.Image:
    square = pad_to_square(image.convert("RGB"))
    return square.resize((input_size, input_size), Image.Resampling.BILINEAR)


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

    # Build thresholds from actual scores only (not user-specified operating points).
    # Operating points are report-at values, not candidates for EER/balanced-accuracy.
    # The sentinels ensure FAR=0.0 and FRR=0.0 are reachable.
    score_thresholds = list(pos) + list(neg)
    score_thresholds.append(float(neg.max()) + 1e-6)
    score_thresholds.append(float(pos.min()) - 1e-6)
    thresholds = sorted(set(score_thresholds))

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
