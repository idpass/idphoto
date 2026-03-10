#!/usr/bin/env python3
"""Parameter sweep over compression settings on LFW pairs.

Evaluates a matrix of compression configs and reports per-model metrics.
Supports checkpoint/resume: if the output JSON already contains completed
configs, they are skipped on restart.

Sweep v2 focuses on:
- Aligned (YuNet) vs bbox crop strategies
- Quality floors (minimum quality, may exceed byte budget)
- Face margin variations for bbox crop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import idphoto
except ImportError as exc:
    raise SystemExit(
        "Could not import 'idphoto'. Build/install bindings first:\n"
        "  maturin develop --manifest-path bindings/python/Cargo.toml"
    ) from exc

import numpy as np
import onnxruntime as ort
from PIL import Image

from arcface_qr_eval import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, FaceAligner
from eval_utils import (
    compute_metrics,
    decode_rgb,
    ensure_model,
    image_to_model_tensor,
    image_to_png_bytes,
    l2_normalize,
    parse_providers,
)

DEFAULT_OFFICIAL_MODEL_PATH = (
    Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
)
DEFAULT_FAR_CAPS = [0.01, 0.001, 0.0001]
DEFAULT_OPERATING_POINTS = [0.10, 0.15, 0.20, 0.25, 0.30]

# Sweep v2 parameter space (all configs use face-bbox for speed;
# gallery alignment is handled by InsightFace aligner at embed time)
BUDGETS = [600, 1000]
DIMENSIONS = [48, 64]
GRAYSCALES = [True, False]
QUALITY_FLOORS = [None, 0.3, 0.5]  # None = no floor (standard compress_to_fit)
DEFAULT_MARGIN = 1.3
BBOX_MARGINS = [1.0, 1.3, 1.8]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LFW compression parameter sweep v2.")
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--providers", default="CPUExecutionProvider")
    p.add_argument(
        "--output", type=Path, default=Path("results/sweep_lfw_v2.json"),
    )
    p.add_argument(
        "--models", nargs="+", default=["int8", "official"],
        choices=["int8", "official"],
        help="Which embedding models to use (default: both)",
    )
    return p.parse_args(argv)


def config_key(
    crop_mode: str,
    budget: int,
    dim: int,
    grayscale: bool,
    quality_floor: Optional[float] = None,
    face_margin: Optional[float] = None,
) -> str:
    """Stable string key for checkpoint lookup."""
    gray_label = "gray" if grayscale else "color"
    parts = [crop_mode, str(budget), str(dim), gray_label]
    if quality_floor is not None:
        parts.append("qf{}".format(quality_floor))
    if face_margin is not None:
        parts.append("m{}".format(face_margin))
    return "_".join(parts)


def build_config_list() -> List[Dict[str, Any]]:
    """Build the full list of configs to sweep.

    All configs use face-bbox crop (rustface detection + bbox crop, no
    tract/YuNet alignment). Gallery images are aligned by InsightFace at
    embed time, so query-side alignment is not needed here.

    Main sweep (24): 2 budgets x 2 dims x 2 grays x 3 quality floors,
    margin fixed at 1.3.
    Margin sweep (3): 3 margins, fixed budget=600, dim=48, gray=True,
    qfloor=None.
    Total: 27 configs.
    """
    configs: List[Dict[str, Any]] = []

    # Main sweep: vary budget, dim, grayscale, quality floor
    for budget in BUDGETS:
        for dim in DIMENSIONS:
            for grayscale in GRAYSCALES:
                for qfloor in QUALITY_FLOORS:
                    configs.append({
                        "crop_mode": "face-bbox",
                        "budget": budget,
                        "dim": dim,
                        "grayscale": grayscale,
                        "quality_floor": qfloor,
                        "face_margin": DEFAULT_MARGIN,
                    })

    # Margin sweep: vary face_margin (skip 1.3, already in main sweep)
    for margin in BBOX_MARGINS:
        if margin == DEFAULT_MARGIN:
            continue
        configs.append({
            "crop_mode": "face-bbox",
            "budget": 600,
            "dim": 48,
            "grayscale": True,
            "quality_floor": None,
            "face_margin": margin,
        })

    return configs


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load existing results or return empty structure."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "configs" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"meta": {}, "baseline": None, "configs": []}


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write: write to temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prepare_pairs(
    max_pairs: int, seed: int,
) -> Tuple[List[Tuple[str, str, bool]], Dict[str, bytes]]:
    """Load LFW and return balanced pair entries + raw image bytes."""
    from sklearn.datasets import fetch_lfw_pairs

    dataset = fetch_lfw_pairs(
        subset="test", funneled=True, resize=1.0, color=True,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    pairs = dataset.pairs
    labels = dataset.target.astype(bool)
    print("LFW loaded: {} pairs, shape={}".format(len(pairs), pairs.shape), flush=True)

    import hashlib
    import random

    positives: List[int] = [i for i, l in enumerate(labels) if l]
    negatives: List[int] = [i for i, l in enumerate(labels) if not l]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_pos = max_pairs // 2
    n_neg = max_pairs - n_pos
    selected_indices = positives[:n_pos] + negatives[:n_neg]

    pair_entries: List[Tuple[str, str, bool]] = []
    raw_by_key: Dict[str, bytes] = {}

    for idx in selected_indices:
        raw_a = image_to_png_bytes(pairs[idx][0])
        raw_b = image_to_png_bytes(pairs[idx][1])
        key_a = hashlib.sha1(raw_a).hexdigest()
        key_b = hashlib.sha1(raw_b).hexdigest()
        raw_by_key.setdefault(key_a, raw_a)
        raw_by_key.setdefault(key_b, raw_b)
        pair_entries.append((key_a, key_b, bool(labels[idx])))

    print(
        "Selected {} pairs (pos={}, neg={}), unique_images={}".format(
            len(pair_entries), n_pos, n_neg, len(raw_by_key),
        ),
        flush=True,
    )
    return pair_entries, raw_by_key


def embed_originals(
    raw_by_key: Dict[str, bytes],
    aligner: FaceAligner,
    int8_session: Optional[ort.InferenceSession],
    int8_input_name: Optional[str],
    int8_input_type: Optional[str],
    official_model: Any,
    model_names: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Embed all original images once. Returns {key: {model: emb}}."""
    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    ordered = sorted(raw_by_key)
    for i, key in enumerate(ordered, start=1):
        img = decode_rgb(raw_by_key[key])
        rgb, _ = aligner.align_to_rgb(img)
        emb: Dict[str, np.ndarray] = {}

        if "int8" in model_names:
            tensor = image_to_model_tensor(rgb, int8_input_type, "none", "rgb")
            int8_out = int8_session.run(None, {int8_input_name: tensor})[0]
            emb["int8"] = l2_normalize(np.asarray(int8_out, dtype=np.float32).reshape(-1))

        if "official" in model_names:
            bgr = rgb[..., ::-1]
            off_out = np.asarray(official_model.get_feat(bgr), dtype=np.float32)
            emb["official"] = l2_normalize(off_out.reshape(-1))

        embeddings[key] = emb
        if i % 100 == 0:
            print("  Original embeddings: {}/{}".format(i, len(ordered)), flush=True)

    return embeddings


def run_one_config(
    precropped_by_key: Dict[str, bytes],
    pair_entries: List[Tuple[str, str, bool]],
    orig_embeddings: Dict[str, Dict[str, np.ndarray]],
    aligner: FaceAligner,
    int8_session: Optional[ort.InferenceSession],
    int8_input_name: Optional[str],
    int8_input_type: Optional[str],
    official_model: Any,
    model_names: List[str],
    budget: int,
    dim: int,
    grayscale: bool,
    quality_floor: Optional[float] = None,
) -> Dict[str, Any]:
    """Compress pre-cropped images, embed, score, and compute metrics."""
    from statistics import mean

    ordered = sorted(precropped_by_key)
    compressed_by_key: Dict[str, bytes] = {}
    sizes: List[int] = []
    qualities: List[float] = []
    reached = 0
    floor_applied = 0

    # Images are already face-cropped; use crop_mode="none" to just resize+encode.
    compress_kwargs: Dict[str, Any] = {
        "preset": "qr-code-match",
        "output_format": "webp",
        "grayscale": grayscale,
        "max_dimension": dim,
        "crop_mode": "none",
    }

    for key in ordered:
        fit = idphoto.compress_to_fit(
            precropped_by_key[key],
            max_bytes=budget,
            **compress_kwargs,
        )
        quality_used = float(fit.quality_used)
        data = bytes(fit.data)
        hit_target = bool(fit.reached_target)

        # Apply quality floor: if quality dropped below the floor, re-compress
        # at the floor quality (may exceed byte budget).
        if quality_floor is not None and quality_used < quality_floor:
            recompressed = idphoto.compress(
                precropped_by_key[key],
                quality=quality_floor,
                **compress_kwargs,
            )
            data = bytes(recompressed.data)
            quality_used = quality_floor
            hit_target = len(data) <= budget
            floor_applied += 1

        compressed_by_key[key] = data
        sizes.append(len(data))
        qualities.append(quality_used)
        if hit_target:
            reached += 1

    comp_stats = {
        "mean_bytes": mean(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "reached_target": reached,
        "total": len(ordered),
        "mean_quality": mean(qualities),
        "floor_applied": floor_applied,
    }

    # Embed compressed images
    comp_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for key in ordered:
        img = decode_rgb(compressed_by_key[key])
        rgb, _ = aligner.align_to_rgb(img)
        emb: Dict[str, np.ndarray] = {}

        if "int8" in model_names:
            tensor = image_to_model_tensor(rgb, int8_input_type, "none", "rgb")
            int8_out = int8_session.run(None, {int8_input_name: tensor})[0]
            emb["int8"] = l2_normalize(np.asarray(int8_out, dtype=np.float32).reshape(-1))

        if "official" in model_names:
            bgr = rgb[..., ::-1]
            off_out = np.asarray(official_model.get_feat(bgr), dtype=np.float32)
            emb["official"] = l2_normalize(off_out.reshape(-1))

        comp_embeddings[key] = emb

    # Score pairs
    model_results: Dict[str, Any] = {}
    for model_name in model_names:
        positives: List[float] = []
        negatives: List[float] = []

        for key_a, key_b, is_same in pair_entries:
            comp_a = comp_embeddings[key_a][model_name]
            orig_b = orig_embeddings[key_b][model_name]
            score_ab = float(np.dot(comp_a, orig_b))

            comp_b = comp_embeddings[key_b][model_name]
            orig_a = orig_embeddings[key_a][model_name]
            score_ba = float(np.dot(comp_b, orig_a))

            avg = (score_ab + score_ba) / 2.0
            if is_same:
                positives.append(avg)
            else:
                negatives.append(avg)

        metrics = compute_metrics(
            positive_scores=positives,
            negative_scores=negatives,
            far_caps=DEFAULT_FAR_CAPS,
            operating_points=DEFAULT_OPERATING_POINTS,
        )
        model_results[model_name] = metrics

    return {
        "compression_stats": comp_stats,
        "models": model_results,
    }


def run_baseline(
    pair_entries: List[Tuple[str, str, bool]],
    orig_embeddings: Dict[str, Dict[str, np.ndarray]],
    model_names: List[str],
) -> Dict[str, Any]:
    """Score original vs original (no compression)."""
    model_results: Dict[str, Any] = {}
    for model_name in model_names:
        positives: List[float] = []
        negatives: List[float] = []

        for key_a, key_b, is_same in pair_entries:
            emb_a = orig_embeddings[key_a][model_name]
            emb_b = orig_embeddings[key_b][model_name]
            score = float(np.dot(emb_a, emb_b))
            if is_same:
                positives.append(score)
            else:
                negatives.append(score)

        metrics = compute_metrics(
            positive_scores=positives,
            negative_scores=negatives,
            far_caps=DEFAULT_FAR_CAPS,
            operating_points=DEFAULT_OPERATING_POINTS,
        )
        model_results[model_name] = metrics

    return {"models": model_results}


def print_summary(data: Dict[str, Any]) -> None:
    """Print a compact summary table of all configs."""
    baseline = data.get("baseline")
    configs = data.get("configs", [])

    # Detect which models are present
    sample_models = {}
    if baseline:
        sample_models = baseline.get("models", {})
    elif configs:
        sample_models = configs[0].get("models", {})
    model_names = sorted(sample_models.keys())

    # Build header dynamically
    parts = ["{:<45s}".format("Config")]
    for m in model_names:
        label = "int8" if m == "int8" else "IF"
        parts.append("{:>9s} {:>9s} {:>11s}".format(
            label + " EER", label + " AUC", label + " TAR@1%",
        ))
    parts.append("{:>6s}".format("bytes"))
    header = " ".join(parts)

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    def row(label: str, models: Dict, extra: str = "") -> str:
        parts = ["{:<45s}".format(label)]
        for m in model_names:
            md = models.get(m, {})
            eer = md.get("eer_approx", {}).get("eer", 0)
            auc = md.get("auc", 0)
            tar_d = md.get("best_under_far", {}).get("0.01", {})
            tar_v = tar_d["tar"] if tar_d else 0
            parts.append("{:>9.4f} {:>9.4f} {:>11.4f}".format(eer, auc, tar_v))
        parts.append("{:>6s}".format(extra))
        return " ".join(parts)

    if baseline:
        print(row("baseline (orig vs orig)", baseline["models"]))
        print("-" * len(header))

    for cfg in sorted(configs, key=lambda c: (
        c["key"]["crop_mode"],
        c["key"]["budget"],
        c["key"]["dim"],
        c["key"]["grayscale"],
        str(c["key"].get("quality_floor", "")),
        str(c["key"].get("face_margin", "")),
    )):
        k = cfg["key"]
        gray_label = "gray" if k["grayscale"] else "color"
        qf = k.get("quality_floor")
        fm = k.get("face_margin")
        label = "{:<15s} {}B {}px {}".format(
            k["crop_mode"], k["budget"], k["dim"], gray_label,
        )
        if qf is not None:
            label += " qf={}".format(qf)
        if fm is not None:
            label += " m={}".format(fm)
        mean_bytes = "{:.0f}".format(cfg["compression_stats"]["mean_bytes"])
        print(row(label, cfg["models"], mean_bytes))

    print("=" * len(header))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    providers = parse_providers(args.providers)

    # Load checkpoint
    data = load_checkpoint(args.output)
    completed_keys = {
        cfg["key_str"] for cfg in data.get("configs", []) if "key_str" in cfg
    }
    baseline_done = data.get("baseline") is not None

    all_configs = build_config_list()
    total_configs = len(all_configs)
    remaining = total_configs - len(completed_keys) + (0 if baseline_done else 1)
    print(
        "Sweep v2: {} configs + baseline. "
        "Already completed: {} configs{}. "
        "Remaining: {}.".format(
            total_configs,
            len(completed_keys),
            " + baseline" if baseline_done else "",
            remaining,
        ),
        flush=True,
    )

    if remaining == 0:
        print("All configs already completed. Printing summary.", flush=True)
        print_summary(data)
        return 0

    # Prepare dataset
    print("\n--- Preparing dataset ---", flush=True)
    pair_entries, raw_by_key = prepare_pairs(args.max_pairs, args.seed)

    # Load models
    model_names = args.models
    print("\n--- Loading models: {} ---".format(", ".join(model_names)), flush=True)

    int8_session = None
    int8_input_name = None
    int8_input_type = None
    if "int8" in model_names:
        ensure_model(DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, True)
        int8_session = ort.InferenceSession(str(DEFAULT_MODEL_PATH), providers=list(providers))
        int8_input = int8_session.get_inputs()[0]
        int8_input_name = int8_input.name
        int8_input_type = int8_input.type

    official_model = None
    if "official" in model_names:
        import insightface
        official_model = insightface.model_zoo.get_model(
            str(DEFAULT_OFFICIAL_MODEL_PATH), providers=list(providers),
        )
        ctx_id = 0 if any(p.startswith("CUDA") for p in providers) else -1
        official_model.prepare(ctx_id=ctx_id)

    aligner = FaceAligner(
        backend="none", providers=providers, input_size=112,
        det_size=640, fail_policy="fallback",
    )

    # Pre-crop images once per unique margin (avoids re-running face detection
    # for every config). Each margin produces a dict of {key: cropped_png_bytes}.
    # Configs then use crop_mode="none" on the pre-cropped data.
    unique_margins = sorted({cfg["face_margin"] for cfg in all_configs if cfg["face_margin"] is not None})
    precropped: Dict[float, Dict[str, bytes]] = {}
    print("\n--- Pre-cropping faces ({} margins) ---".format(len(unique_margins)), flush=True)
    for margin in unique_margins:
        t0 = time.time()
        cropped: Dict[str, bytes] = {}
        for key in sorted(raw_by_key):
            result = idphoto.compress(
                raw_by_key[key],
                crop_mode="face-bbox",
                face_margin=margin,
                max_dimension=512,  # large enough to preserve detail
                quality=1.0,
                output_format="webp",
            )
            cropped[key] = bytes(result.data)
        elapsed = time.time() - t0
        precropped[margin] = cropped
        print("  margin={}: {:.1f}s ({} images)".format(margin, elapsed, len(cropped)), flush=True)

    # Compute original embeddings (once)
    print("\n--- Computing original embeddings ---", flush=True)
    orig_embeddings = embed_originals(
        raw_by_key, aligner,
        int8_session, int8_input_name, int8_input_type,
        official_model, model_names,
    )

    # Update meta
    data["meta"] = {
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "pair_count": len(pair_entries),
        "unique_images": len(raw_by_key),
        "dataset": "lfw",
        "lfw_slice": "full",
        "format": "webp",
        "preset": "qr-code-match",
        "sweep_version": 2,
        "budgets": BUDGETS,
        "dimensions": DIMENSIONS,
        "grayscales": GRAYSCALES,
        "quality_floors": [x if x is not None else "none" for x in QUALITY_FLOORS],
        "default_margin": DEFAULT_MARGIN,
        "bbox_margins": BBOX_MARGINS,
        "models": model_names,
        "far_caps": DEFAULT_FAR_CAPS,
        "started": data.get("meta", {}).get("started", datetime.now().isoformat()),
    }

    # Baseline
    if not baseline_done:
        print("\n--- Running baseline (original vs original) ---", flush=True)
        t0 = time.time()
        data["baseline"] = run_baseline(pair_entries, orig_embeddings, model_names)
        elapsed = time.time() - t0
        print("  Baseline done in {:.1f}s".format(elapsed), flush=True)
        save_checkpoint(args.output, data)

    # Sweep
    for i, cfg in enumerate(all_configs, start=1):
        key_str = config_key(
            cfg["crop_mode"], cfg["budget"], cfg["dim"], cfg["grayscale"],
            cfg["quality_floor"], cfg["face_margin"],
        )
        if key_str in completed_keys:
            continue

        gray_label = "gray" if cfg["grayscale"] else "color"
        extras = ""
        if cfg["quality_floor"] is not None:
            extras += " qf={}".format(cfg["quality_floor"])
        if cfg["face_margin"] is not None:
            extras += " m={}".format(cfg["face_margin"])
        print(
            "\n--- Config {}/{}: {} {}B {}px {}{}---".format(
                i, total_configs, cfg["crop_mode"], cfg["budget"],
                cfg["dim"], gray_label, extras + " " if extras else "",
            ),
            flush=True,
        )
        t0 = time.time()

        margin = cfg["face_margin"] or DEFAULT_MARGIN
        result = run_one_config(
            precropped[margin], pair_entries, orig_embeddings,
            aligner, int8_session, int8_input_name, int8_input_type,
            official_model, model_names,
            budget=cfg["budget"],
            dim=cfg["dim"],
            grayscale=cfg["grayscale"],
            quality_floor=cfg["quality_floor"],
        )

        elapsed = time.time() - t0
        result["key"] = {
            "crop_mode": cfg["crop_mode"],
            "budget": cfg["budget"],
            "dim": cfg["dim"],
            "grayscale": cfg["grayscale"],
            "quality_floor": cfg["quality_floor"],
            "face_margin": cfg["face_margin"],
        }
        result["key_str"] = key_str
        result["elapsed_s"] = round(elapsed, 1)

        data["configs"].append(result)
        completed_keys.add(key_str)
        save_checkpoint(args.output, data)

        stats = result["compression_stats"]
        eer_parts = []
        for mn in model_names:
            eer = result["models"][mn]["eer_approx"]["eer"]
            eer_parts.append("{}_EER={:.4f}".format(mn, eer))
        print(
            "  Done in {:.1f}s. mean_bytes={:.0f} {}".format(
                elapsed, stats["mean_bytes"], " ".join(eer_parts),
            ),
            flush=True,
        )

    data["meta"]["completed"] = datetime.now().isoformat()
    save_checkpoint(args.output, data)

    print_summary(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
