from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import write_json
from .metrics import compute_asr_metrics


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def utterance_wer(reference: str, hypothesis: str) -> float:
    return compute_asr_metrics([reference], [hypothesis]).wer


def score_rows(rows: list[dict], mode: str, seed: int) -> np.ndarray:
    if mode == "peak_entropy":
        return np.asarray([r["fast_confidence"] for r in rows], dtype=np.float64)
    if mode == "peak":
        return np.asarray([r["fast_confidence"] + 0.03 * r["fast_entropy"] for r in rows], dtype=np.float64)
    if mode == "entropy":
        return -np.asarray([r["fast_entropy"] for r in rows], dtype=np.float64)
    if mode == "duration":
        return -np.asarray([r["duration"] for r in rows], dtype=np.float64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.random(len(rows))
    if mode == "oracle":
        gains = []
        for r in rows:
            fast = utterance_wer(r["reference"], r["fast_hypothesis"])
            strong = utterance_wer(r["reference"], r["strong_hypothesis"])
            gains.append(fast - strong)
        # Higher gain should be routed first, so invert after sorting ascending.
        return -np.asarray(gains, dtype=np.float64)
    raise ValueError(f"Unsupported routing mode: {mode}")


def evaluate_mode(rows: list[dict], scores: np.ndarray, budgets: list[float]) -> list[dict]:
    refs = [r["reference"] for r in rows]
    audio_seconds = sum(r["duration"] for r in rows)
    fast_decode_seconds = sum(r["fast_decode_seconds"] for r in rows)
    order = np.argsort(scores)
    out = []
    for budget in budgets:
        routed_n = int(round(len(rows) * budget / 100.0))
        routed = set(order[:routed_n].tolist())
        hyps = [r["strong_hypothesis"] if i in routed else r["fast_hypothesis"] for i, r in enumerate(rows)]
        metrics = compute_asr_metrics(refs, hyps)
        strong_seconds = sum(rows[i]["strong_decode_seconds"] for i in routed)
        out.append(
            {
                "route_percentage": budget,
                "routed_utterances": routed_n,
                "wer": metrics.wer,
                "cer": metrics.cer,
                "rtf": (fast_decode_seconds + strong_seconds) / max(audio_seconds, 1e-8),
                "substitutions": metrics.substitutions,
                "deletions": metrics.deletions,
                "insertions": metrics.insertions,
            }
        )
    return out


def complementarity(rows: list[dict]) -> dict:
    fast_better = strong_better = tied = 0
    for r in rows:
        fast = utterance_wer(r["reference"], r["fast_hypothesis"])
        strong = utterance_wer(r["reference"], r["strong_hypothesis"])
        if fast < strong:
            fast_better += 1
        elif strong < fast:
            strong_better += 1
        else:
            tied += 1
    return {
        "fast_better": fast_better,
        "strong_better": strong_better,
        "tied": tied,
        "n": len(rows),
    }


def main() -> None:
    args = build_argparser().parse_args()
    rows = read_jsonl(args.predictions)
    all_modes = {}
    for mode in args.modes:
        mode_runs = []
        seeds = args.random_seeds if mode == "random" else [args.seed]
        for seed in seeds:
            scores = score_rows(rows, mode, seed)
            mode_runs.append({"seed": seed, "budgets": evaluate_mode(rows, scores, args.budgets)})
        all_modes[mode] = mode_runs

    result = {
        "predictions": args.predictions,
        "n": len(rows),
        "budgets": args.budgets,
        "modes": all_modes,
        "complementarity": complementarity(rows),
    }
    write_json(args.output, result)
    print(result)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablate SpeechMaster routing scores from saved predictions.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budgets", type=float, nargs="+", default=[0, 10, 25, 50, 75, 100])
    p.add_argument(
        "--modes",
        nargs="+",
        default=["peak_entropy", "peak", "entropy", "duration", "random", "oracle"],
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--random-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return p


if __name__ == "__main__":
    main()

