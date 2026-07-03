from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import write_json
from .metrics import compute_asr_metrics


FEATURE_GROUPS = {
    "ctc_only": [
        "fast_confidence",
        "fast_entropy",
        "fast_peak",
        "confidence_risk",
        "peak_risk",
    ],
    "duration_only": [
        "duration",
        "log_duration",
        "fast_word_count",
        "fast_char_count",
    ],
    "ctc_duration": [
        "fast_confidence",
        "fast_entropy",
        "fast_peak",
        "confidence_risk",
        "peak_risk",
        "duration",
        "log_duration",
        "entropy_duration",
        "peak_risk_duration",
        "confidence_risk_duration",
    ],
    "full": [
        "fast_confidence",
        "fast_entropy",
        "fast_peak",
        "confidence_risk",
        "peak_risk",
        "duration",
        "log_duration",
        "fast_word_count",
        "fast_char_count",
        "fast_word_rate",
        "fast_char_rate",
        "entropy_duration",
        "peak_risk_duration",
        "confidence_risk_duration",
    ],
}


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def edit_count(reference: str, hypothesis: str) -> int:
    metrics = compute_asr_metrics([reference], [hypothesis])
    return metrics.substitutions + metrics.deletions + metrics.insertions


def feature_dict(row: dict) -> dict[str, float]:
    duration = max(float(row.get("duration") or 0.0), 1e-3)
    hypothesis = row.get("fast_hypothesis") or ""
    word_count = len(hypothesis.split())
    char_count = len(hypothesis.replace(" ", ""))
    confidence = float(row.get("fast_confidence") or 0.0)
    entropy = float(row.get("fast_entropy") or 0.0)
    peak = confidence + 0.03 * entropy
    confidence_risk = 1.0 - confidence
    peak_risk = 1.0 - peak
    return {
        "fast_confidence": confidence,
        "fast_entropy": entropy,
        "fast_peak": peak,
        "confidence_risk": confidence_risk,
        "peak_risk": peak_risk,
        "duration": duration,
        "log_duration": math.log1p(duration),
        "fast_word_count": float(word_count),
        "fast_char_count": float(char_count),
        "fast_word_rate": word_count / duration,
        "fast_char_rate": char_count / duration,
        "entropy_duration": entropy * duration,
        "peak_risk_duration": peak_risk * duration,
        "confidence_risk_duration": confidence_risk * duration,
    }


def build_features(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[feature_dict(row)[name] for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def gain_targets(rows: list[dict]) -> np.ndarray:
    gains = []
    for row in rows:
        fast_errors = edit_count(row["reference"], row["fast_hypothesis"])
        strong_errors = edit_count(row["reference"], row["strong_hypothesis"])
        gains.append(float(fast_errors - strong_errors))
    return np.asarray(gains, dtype=np.float64)


def fit_gain_router(rows: list[dict], feature_names: list[str], alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    targets = gain_targets(rows)
    model.fit(build_features(rows, feature_names), targets)
    return model, targets


def evaluate_budgets(rows: list[dict], scores: np.ndarray, budgets: list[float]) -> list[dict]:
    refs = [row["reference"] for row in rows]
    audio_seconds = sum(float(row.get("duration") or 0.0) for row in rows)
    fast_decode_seconds = sum(float(row.get("fast_decode_seconds") or 0.0) for row in rows)
    order = np.argsort(-scores)
    out = []
    for budget in budgets:
        routed_n = int(round(len(rows) * budget / 100.0))
        routed = set(order[:routed_n].tolist())
        hyps = [
            row["strong_hypothesis"] if index in routed else row["fast_hypothesis"]
            for index, row in enumerate(rows)
        ]
        metrics = compute_asr_metrics(refs, hyps)
        routed_audio = sum(float(rows[index].get("duration") or 0.0) for index in routed)
        strong_decode_seconds = sum(
            float(rows[index].get("strong_decode_seconds") or 0.0) for index in routed
        )
        out.append(
            {
                "route_percentage": budget,
                "routed_utterances": routed_n,
                "routed_audio_seconds": routed_audio,
                "wer": metrics.wer,
                "cer": metrics.cer,
                "rtf": (fast_decode_seconds + strong_decode_seconds) / max(audio_seconds, 1e-8),
                "substitutions": metrics.substitutions,
                "deletions": metrics.deletions,
                "insertions": metrics.insertions,
                "mean_predicted_gain": float(np.mean(scores[order[:routed_n]])) if routed_n else 0.0,
            }
        )
    return out


def oracle_metrics(rows: list[dict]) -> dict:
    refs = [row["reference"] for row in rows]
    hyps = []
    fast_better = strong_better = tied = 0
    total_positive_gain = 0.0
    for row in rows:
        fast_errors = edit_count(row["reference"], row["fast_hypothesis"])
        strong_errors = edit_count(row["reference"], row["strong_hypothesis"])
        gain = fast_errors - strong_errors
        if gain > 0:
            strong_better += 1
            total_positive_gain += gain
            hyps.append(row["strong_hypothesis"])
        elif gain < 0:
            fast_better += 1
            hyps.append(row["fast_hypothesis"])
        else:
            tied += 1
            hyps.append(row["fast_hypothesis"])
    metrics = compute_asr_metrics(refs, hyps)
    return {
        "wer": metrics.wer,
        "cer": metrics.cer,
        "fast_better": fast_better,
        "strong_better": strong_better,
        "tied": tied,
        "n": len(rows),
        "positive_gain_edit_count": total_positive_gain,
    }


def model_summary(model, feature_names: list[str]) -> dict:
    ridge = model.named_steps["ridge"]
    return {
        "feature_names": feature_names,
        "standardized_coefficients": {
            name: float(coef) for name, coef in zip(feature_names, ridge.coef_)
        },
        "intercept": float(ridge.intercept_),
    }


def evaluate_car(args: argparse.Namespace) -> dict:
    train_rows = read_jsonl(args.train_predictions)
    eval_rows = read_jsonl(args.eval_predictions)
    feature_names = FEATURE_GROUPS[args.feature_group]
    model, train_targets = fit_gain_router(train_rows, feature_names, args.alpha)
    eval_scores = model.predict(build_features(eval_rows, feature_names))
    oracle = oracle_metrics(eval_rows)

    feature_ablations = {}
    for group, names in FEATURE_GROUPS.items():
        ablation_model, _ = fit_gain_router(train_rows, names, args.alpha)
        scores = ablation_model.predict(build_features(eval_rows, names))
        feature_ablations[group] = evaluate_budgets(eval_rows, scores, args.ablation_budgets)

    result = {
        "system": "SpeechMaster-CAR",
        "router": "complementarity_aware_gain_regression",
        "train_predictions": args.train_predictions,
        "predictions": args.eval_predictions,
        "n": len(eval_rows),
        "train_n": len(train_rows),
        "feature_group": args.feature_group,
        "alpha": args.alpha,
        "budgets": evaluate_budgets(eval_rows, eval_scores, args.budgets),
        "oracle_wer": oracle["wer"],
        "oracle_cer": oracle["cer"],
        "complementarity": {
            "fast_better": oracle["fast_better"],
            "strong_better": oracle["strong_better"],
            "tied": oracle["tied"],
            "n": oracle["n"],
            "positive_gain_edit_count": oracle["positive_gain_edit_count"],
        },
        "train_target": {
            "positive_count": int(np.sum(train_targets > 0)),
            "negative_count": int(np.sum(train_targets < 0)),
            "tied_count": int(np.sum(train_targets == 0)),
            "mean_gain": float(np.mean(train_targets)),
            "sum_gain": float(np.sum(train_targets)),
        },
        "model": model_summary(model, feature_names),
        "feature_ablations": feature_ablations,
    }
    write_json(args.output, result)
    print(result)
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train SpeechMaster-CAR from cached fast/strong branch predictions "
            "and evaluate budgeted complementarity-aware routing."
        )
    )
    p.add_argument("--train-predictions", required=True)
    p.add_argument("--eval-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature-group", choices=sorted(FEATURE_GROUPS), default="full")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--budgets", type=float, nargs="+", default=[0, 10, 25, 50, 75, 100])
    p.add_argument("--ablation-budgets", type=float, nargs="+", default=[10, 25, 50, 75])
    return p


def main() -> None:
    evaluate_car(build_argparser().parse_args())


if __name__ == "__main__":
    main()
