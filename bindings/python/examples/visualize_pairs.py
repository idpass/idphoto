#!/usr/bin/env python3
"""Generate an HTML page comparing crop strategies on LFW pairs.

For each pair, shows both crop modes side-by-side with clear PASS/FAIL
indicators, so you can see at a glance which strategy works better.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sklearn.datasets import fetch_lfw_pairs

from arcface_qr_eval import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, FaceAligner
from eval_utils import (
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

# Threshold for pass/fail display (InsightFace EER threshold from previous eval)
DEFAULT_THRESHOLD = 0.13


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare crop strategies on LFW pairs.")
    p.add_argument("--max-pairs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--providers", default="CPUExecutionProvider")
    p.add_argument("--preset", default="qr-code-match")
    p.add_argument("--max-bytes", type=int, default=600)
    p.add_argument("--output-format", default="webp", choices=["jpeg", "webp"])
    p.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-dimension", type=int, default=48)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Cosine similarity threshold for pass/fail display.")
    p.add_argument("--model", default="official", choices=["official", "int8", "both"],
                   help="Which model to score with.")
    p.add_argument(
        "--output", type=Path, default=Path("results/pair_comparison.html"),
    )
    return p.parse_args(argv)


def to_data_uri(data: bytes) -> str:
    if data[:4] == b"RIFF":
        mime = "image/webp"
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def array_to_pil(arr: np.ndarray) -> Image.Image:
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    providers = parse_providers(args.providers)
    threshold = args.threshold

    # Load LFW full-res
    dataset = fetch_lfw_pairs(
        subset="test", funneled=True, resize=1.0, color=True,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    pairs = dataset.pairs
    labels = dataset.target.astype(bool)
    print(f"LFW loaded: {len(pairs)} pairs", flush=True)

    # Balance and select pairs
    import random
    positives = [(i, True) for i, l in enumerate(labels) if l]
    negatives = [(i, False) for i, l in enumerate(labels) if not l]
    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_pos = args.max_pairs // 2
    n_neg = args.max_pairs - n_pos
    selected = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(selected)
    print(f"Selected {len(selected)} pairs (pos={n_pos}, neg={n_neg})", flush=True)

    # Load models
    models: Dict[str, object] = {}

    if args.model in ("int8", "both"):
        ensure_model(DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, True)
        int8_session = ort.InferenceSession(str(DEFAULT_MODEL_PATH), providers=list(providers))
        int8_input = int8_session.get_inputs()[0]
        models["int8 (arcfaceresnet100)"] = ("int8", int8_session, int8_input)
        print("Loaded int8 model", flush=True)

    if args.model in ("official", "both"):
        official_path = DEFAULT_OFFICIAL_MODEL_PATH
        if official_path.exists():
            import insightface
            official_model = insightface.model_zoo.get_model(
                str(official_path), providers=list(providers),
            )
            ctx_id = 0 if any(p.startswith("CUDA") for p in providers) else -1
            official_model.prepare(ctx_id=ctx_id)
            models["InsightFace w600k_r50"] = ("official", official_model, None)
            print("Loaded InsightFace official model", flush=True)
        else:
            print(f"InsightFace model not found at {official_path}, skipping", flush=True)

    if not models:
        print("No models available. Exiting.", flush=True)
        return 1

    aligner = FaceAligner(
        backend="none", providers=providers, input_size=112,
        det_size=640, fail_policy="fallback",
    )

    def embed_with(model_key: str, rgb: np.ndarray) -> np.ndarray:
        kind, model_obj, input_meta = models[model_key]
        if kind == "official":
            bgr = rgb[..., ::-1]
            out = np.asarray(model_obj.get_feat(bgr), dtype=np.float32)
            return l2_normalize(out.reshape(-1))
        else:
            tensor = image_to_model_tensor(rgb, input_meta.type, "none", "rgb")
            out = model_obj.run(None, {input_meta.name: tensor})[0]
            return l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))

    # The two crop strategies to compare
    strategies = [
        ("No crop", "none"),
        ("Face detection", "face-detection"),
    ]

    base_kwargs = {
        "preset": args.preset,
        "output_format": args.output_format,
        "grayscale": args.grayscale,
        "max_dimension": args.max_dimension,
    }

    # Process pairs
    model_names = list(models.keys())
    pair_rows: List[str] = []
    # tallies[strat_name][model_name] = {tp, fp, tn, fn}
    tallies: Dict[str, Dict[str, Dict[str, int]]] = {}
    for strat_name, _ in strategies:
        tallies[strat_name] = {}
        for mk in model_names:
            tallies[strat_name][mk] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for pair_idx, (data_idx, is_same) in enumerate(selected, start=1):
        img_a_arr = pairs[data_idx][0]
        img_b_arr = pairs[data_idx][1]
        raw_a = image_to_png_bytes(img_a_arr)
        raw_b = image_to_png_bytes(img_b_arr)

        orig_a = array_to_pil(img_a_arr)
        orig_b = array_to_pil(img_b_arr)

        # Embed originals using idphoto face detection to crop around face,
        # then pad+resize to 112x112 for the embedding model.
        gallery_a = idphoto.compress(
            raw_a, crop_mode="face-detection", max_dimension=112,
            quality=1.0, grayscale=False, output_format="webp",
        )
        gallery_b = idphoto.compress(
            raw_b, crop_mode="face-detection", max_dimension=112,
            quality=1.0, grayscale=False, output_format="webp",
        )
        orig_rgb_a, _ = aligner.align_to_rgb(decode_rgb(bytes(gallery_a.data)))
        orig_rgb_b, _ = aligner.align_to_rgb(decode_rgb(bytes(gallery_b.data)))
        orig_embs = {}
        for mk in model_names:
            orig_embs[mk] = (embed_with(mk, orig_rgb_a), embed_with(mk, orig_rgb_b))

        # For each strategy, compress then score with all models
        strategy_cells: List[str] = []
        for strat_name, crop_mode in strategies:
            fit_a = idphoto.compress_to_fit(
                raw_a, max_bytes=args.max_bytes, crop_mode=crop_mode, **base_kwargs,
            )
            fit_b = idphoto.compress_to_fit(
                raw_b, max_bytes=args.max_bytes, crop_mode=crop_mode, **base_kwargs,
            )
            comp_a = bytes(fit_a.data)
            comp_b = bytes(fit_b.data)

            comp_rgb_a, _ = aligner.align_to_rgb(decode_rgb(comp_a))
            comp_rgb_b, _ = aligner.align_to_rgb(decode_rgb(comp_b))

            face_a_note = ""
            if fit_a.face_bounds is not None:
                face_a_note = '<span class="face-tag">face found</span>'
            face_b_note = ""
            if fit_b.face_bounds is not None:
                face_b_note = '<span class="face-tag">face found</span>'

            # Score with each model
            model_score_rows = []
            any_wrong = False
            for mk in model_names:
                emb_comp_a = embed_with(mk, comp_rgb_a)
                emb_comp_b = embed_with(mk, comp_rgb_b)
                emb_orig_a, emb_orig_b = orig_embs[mk]

                score = (
                    float(np.dot(emb_comp_a, emb_orig_b)) +
                    float(np.dot(emb_comp_b, emb_orig_a))
                ) / 2.0

                predicted_same = score >= threshold
                correct = predicted_same == is_same

                if is_same and predicted_same:
                    tallies[strat_name][mk]["tp"] += 1
                elif is_same and not predicted_same:
                    tallies[strat_name][mk]["fn"] += 1
                elif not is_same and predicted_same:
                    tallies[strat_name][mk]["fp"] += 1
                else:
                    tallies[strat_name][mk]["tn"] += 1

                if not correct:
                    any_wrong = True

                bar_pct = max(0, min(100, score / 0.6 * 100))
                thr_pct = max(0, min(100, threshold / 0.6 * 100))
                bar_color = "#4caf50" if score >= threshold else "#f44336"
                verdict_cls = "correct" if correct else "wrong"
                verdict_text = "CORRECT" if correct else "WRONG"
                # Short model label for display
                short_name = mk.split(" ")[0]

                model_score_rows.append(
                    f'<div class="model-row">'
                    f'<span class="model-label">{short_name}</span>'
                    f'<span class="score-value">{score:.3f}</span>'
                    f'<div class="score-track">'
                    f'<div class="score-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>'
                    f'<div class="score-threshold" style="left:{thr_pct:.1f}%"></div>'
                    f'</div>'
                    f'<span class="mini-verdict {verdict_cls}">{verdict_text}</span>'
                    f'</div>'
                )

            overall_cls = "correct" if not any_wrong else "wrong"
            overall_text = "ALL CORRECT" if not any_wrong else "HAS ERRORS"

            strategy_cells.append(
                f'<div class="strategy">'
                f'<div class="strat-header">{strat_name}</div>'
                f'<div class="strat-images">'
                f'<div class="strat-img">'
                f'<img src="{to_data_uri(comp_a)}">'
                f'<div class="img-meta">{fit_a.width}x{fit_a.height} {len(comp_a)}B {face_a_note}</div>'
                f'</div>'
                f'<div class="strat-img">'
                f'<img src="{to_data_uri(comp_b)}">'
                f'<div class="img-meta">{fit_b.width}x{fit_b.height} {len(comp_b)}B {face_b_note}</div>'
                f'</div>'
                f'</div>'
                f'<div class="model-scores">'
                + "".join(model_score_rows)
                + f'</div>'
                f'<div class="verdict {overall_cls}">{overall_text}</div>'
                f'</div>'
            )

        truth_cls = "match" if is_same else "non-match"
        truth_text = "SAME PERSON" if is_same else "DIFFERENT PEOPLE"

        pair_rows.append(
            f'<div class="pair">'
            f'<div class="pair-left">'
            f'<div class="pair-num">#{pair_idx}</div>'
            f'<div class="originals">'
            f'<img src="{pil_to_data_uri(orig_a)}" class="orig-img">'
            f'<img src="{pil_to_data_uri(orig_b)}" class="orig-img">'
            f'</div>'
            f'<div class="truth {truth_cls}">{truth_text}</div>'
            f'</div>'
            f'<div class="pair-strategies">'
            + "".join(strategy_cells)
            + f'</div>'
            f'</div>'
        )

        if pair_idx % 5 == 0:
            print(f"Processed {pair_idx}/{len(selected)} pairs", flush=True)

    # Summary table
    summary_rows = []
    for strat_name, _ in strategies:
        for mk in model_names:
            t = tallies[strat_name][mk]
            total = t["tp"] + t["fp"] + t["tn"] + t["fn"]
            accuracy = (t["tp"] + t["tn"]) / total if total > 0 else 0
            short_model = mk.split(" ")[0]
            summary_rows.append(
                f'<tr>'
                f'<td class="strat-name">{strat_name}</td>'
                f'<td>{short_model}</td>'
                f'<td class="num">{t["tp"]}</td>'
                f'<td class="num">{t["tn"]}</td>'
                f'<td class="num wrong-cell">{t["fp"]}</td>'
                f'<td class="num wrong-cell">{t["fn"]}</td>'
                f'<td class="num acc">{accuracy:.0%}</td>'
                f'</tr>'
            )

    css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f0f0f0; padding: 24px; max-width: 1200px; margin: 0 auto; }
h1 { margin-bottom: 4px; font-size: 22px; }
.subtitle { color: #666; font-size: 13px; margin-bottom: 16px; }
.summary { background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; border: 1px solid #ddd; }
.summary h2 { font-size: 15px; margin-bottom: 8px; }
.summary table { width: 100%; border-collapse: collapse; font-size: 13px; }
.summary th, .summary td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }
.summary th { background: #f8f8f8; font-weight: 600; }
.summary .strat-name { text-align: left; font-weight: 600; }
.summary .num { font-family: monospace; }
.summary .wrong-cell { color: #c62828; }
.summary .acc { font-weight: 700; font-size: 15px; }

.pair { display: flex; gap: 16px; background: #fff; border-radius: 8px; padding: 14px; margin-bottom: 12px; border: 1px solid #ddd; align-items: stretch; }
.pair-left { flex: 0 0 180px; display: flex; flex-direction: column; align-items: center; gap: 6px; }
.pair-num { font-weight: 700; font-size: 13px; color: #888; }
.originals { display: flex; gap: 4px; }
.orig-img { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc; }
.truth { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-align: center; }
.truth.match { background: #e8f5e9; color: #2e7d32; }
.truth.non-match { background: #fce4ec; color: #c62828; }

.pair-strategies { flex: 1; display: flex; gap: 12px; }
.strategy { flex: 1; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }
.strat-header { font-size: 12px; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
.strat-images { display: flex; gap: 8px; justify-content: center; }
.strat-img { text-align: center; }
.strat-img img { width: 96px; height: 96px; object-fit: contain; image-rendering: pixelated; border: 1px solid #ddd; border-radius: 4px; background: #f8f8f8; }
.img-meta { font-size: 10px; color: #999; margin-top: 2px; }
.face-tag { background: #e3f2fd; color: #1565c0; padding: 1px 4px; border-radius: 2px; font-size: 9px; }
.model-scores { display: flex; flex-direction: column; gap: 4px; }
.model-row { display: grid; grid-template-columns: 70px 45px 1fr 65px; gap: 6px; align-items: center; font-size: 11px; }
.model-label { font-weight: 600; color: #555; }
.score-value { font-family: monospace; font-weight: 600; }
.mini-verdict { font-size: 10px; font-weight: 700; text-align: center; padding: 1px 4px; border-radius: 3px; }
.mini-verdict.correct { background: #e8f5e9; color: #2e7d32; }
.mini-verdict.wrong { background: #ffebee; color: #c62828; }
.score-track { height: 6px; background: #eee; border-radius: 3px; position: relative; margin: 2px 0; }
.score-fill { height: 100%; border-radius: 3px; }
.score-threshold { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #333; border-radius: 1px; }
.verdict { text-align: center; font-size: 13px; font-weight: 800; padding: 4px; border-radius: 4px; }
.verdict.correct { background: #e8f5e9; color: #2e7d32; }
.verdict.wrong { background: #ffebee; color: #c62828; }
"""

    js = """
function filterPairs(mode) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.pair').forEach(el => {
    if (mode === 'all') { el.style.display = ''; return; }
    // Check if any strategy got it wrong
    const verdicts = el.querySelectorAll('.verdict');
    const hasWrong = Array.from(verdicts).some(v => v.classList.contains('wrong'));
    if (mode === 'errors') { el.style.display = hasWrong ? '' : 'none'; }
    else if (mode === 'correct') { el.style.display = hasWrong ? 'none' : ''; }
  });
}
"""

    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
        '<title>Crop Strategy Comparison</title>\n'
        f'<style>{css}</style></head><body>\n'
        '<h1>Crop Strategy Comparison</h1>\n'
        f'<div class="subtitle">Scoring: <b>{", ".join(model_names)}</b> | '
        f'LFW 250x250 | {args.preset} preset | '
        f'{args.max_dimension}px max dim | {args.max_bytes}B budget | '
        f'threshold={threshold:.2f} | {len(selected)} pairs</div>\n'
        '<div class="summary">\n'
        '<h2>Results at threshold ' + f'{threshold:.2f}</h2>\n'
        '<table><tr><th>Strategy</th><th>Model</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Accuracy</th></tr>\n'
        + "".join(summary_rows)
        + '</table></div>\n'
        '<div style="margin:12px 0;display:flex;gap:8px;font-size:13px;">\n'
        '<span>Filter:</span>\n'
        '<button class="filter-btn active" onclick="filterPairs(\'all\')" '
        'style="padding:4px 12px;border:1px solid #ccc;border-radius:4px;background:#333;color:#fff;cursor:pointer">All</button>\n'
        '<button class="filter-btn" onclick="filterPairs(\'errors\')" '
        'style="padding:4px 12px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer">Errors only</button>\n'
        '<button class="filter-btn" onclick="filterPairs(\'correct\')" '
        'style="padding:4px 12px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer">All correct</button>\n'
        '</div>\n'
        + "".join(pair_rows)
        + f'\n<script>{js}</script></body></html>'
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"\nWrote {args.output} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
