from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor

from .data import duration_seconds, load_librispeech_split
from .io import write_json, write_jsonl
from .metrics import compute_asr_metrics


@dataclass
class BranchOutput:
    hypothesis: str
    confidence: float
    entropy: float
    decode_seconds: float


def ctc_decode_with_confidence(model, processor, sample: dict, device: torch.device) -> BranchOutput:
    inputs = processor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_values, attention_mask=attention_mask).logits
    elapsed = time.perf_counter() - start

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    pred_ids = torch.argmax(probs, dim=-1)
    blank_id = getattr(processor.tokenizer, "pad_token_id", None)
    mask = pred_ids.ne(blank_id) if blank_id is not None else torch.ones_like(pred_ids, dtype=torch.bool)
    if mask.any():
        frame_probs = probs[mask]
    else:
        frame_probs = probs
    max_prob = frame_probs.max(dim=-1).values.mean().item()
    entropy = -(frame_probs * torch.log(frame_probs.clamp_min(1e-8))).sum(dim=-1).mean().item()
    hyp = processor.batch_decode(pred_ids.unsqueeze(0).cpu())[0]
    # High confidence should mean peaked non-blank CTC evidence and low entropy.
    confidence = float(max_prob - 0.03 * entropy)
    return BranchOutput(hypothesis=hyp, confidence=confidence, entropy=float(entropy), decode_seconds=elapsed)


def evaluate_speechmaster(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    fast_processor = AutoProcessor.from_pretrained(args.fast_model)
    fast_model = AutoModelForCTC.from_pretrained(args.fast_model).to(device).eval()
    strong_processor = AutoProcessor.from_pretrained(args.strong_model)
    strong_model = AutoModelForCTC.from_pretrained(args.strong_model).to(device).eval()

    samples = load_librispeech_split(
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        streaming=not args.no_streaming,
        cache_dir=args.cache_dir,
    )

    rows = []
    for sample in tqdm(samples, desc="SpeechMaster"):
        fast = ctc_decode_with_confidence(fast_model, fast_processor, sample, device)
        strong = ctc_decode_with_confidence(strong_model, strong_processor, sample, device)
        rows.append(
            {
                "id": sample["id"],
                "reference": sample["text"],
                "duration": duration_seconds(sample),
                "fast_hypothesis": fast.hypothesis,
                "fast_confidence": fast.confidence,
                "fast_entropy": fast.entropy,
                "fast_decode_seconds": fast.decode_seconds,
                "strong_hypothesis": strong.hypothesis,
                "strong_confidence": strong.confidence,
                "strong_entropy": strong.entropy,
                "strong_decode_seconds": strong.decode_seconds,
            }
        )

    refs = [r["reference"] for r in rows]
    audio_seconds = sum(r["duration"] for r in rows)
    fast_decode_seconds = sum(r["fast_decode_seconds"] for r in rows)
    strong_decode_seconds = sum(r["strong_decode_seconds"] for r in rows)

    order = np.argsort([r["fast_confidence"] for r in rows])
    budgets = []
    for pct in args.route_percentages:
        route_n = int(round(len(rows) * pct / 100.0))
        routed = set(order[:route_n].tolist())
        hyps = [r["strong_hypothesis"] if i in routed else r["fast_hypothesis"] for i, r in enumerate(rows)]
        metrics = compute_asr_metrics(refs, hyps)
        routed_audio = sum(rows[i]["duration"] for i in routed)
        routed_strong_time = sum(rows[i]["strong_decode_seconds"] for i in routed)
        total_decode = fast_decode_seconds + routed_strong_time
        budgets.append(
            {
                "route_percentage": pct,
                "routed_utterances": route_n,
                "routed_audio_seconds": routed_audio,
                "wer": metrics.wer,
                "cer": metrics.cer,
                "rtf": total_decode / max(audio_seconds, 1e-8),
                "substitutions": metrics.substitutions,
                "deletions": metrics.deletions,
                "insertions": metrics.insertions,
            }
        )

    # Upper-bound analysis: how much complementarity exists if a perfect router
    # knew which branch has fewer utterance-level word errors.
    oracle_hyps = []
    for r in rows:
        fast_m = compute_asr_metrics([r["reference"]], [r["fast_hypothesis"]])
        strong_m = compute_asr_metrics([r["reference"]], [r["strong_hypothesis"]])
        oracle_hyps.append(r["strong_hypothesis"] if strong_m.wer < fast_m.wer else r["fast_hypothesis"])
    oracle = compute_asr_metrics(refs, oracle_hyps)

    result = {
        "system": "SpeechMaster",
        "fast_model": args.fast_model,
        "strong_model": args.strong_model,
        "split": args.split,
        "subset": args.subset,
        "limit": len(rows),
        "audio_seconds": audio_seconds,
        "fast_decode_seconds": fast_decode_seconds,
        "strong_decode_seconds": strong_decode_seconds,
        "fast_rtf": fast_decode_seconds / max(audio_seconds, 1e-8),
        "strong_rtf": strong_decode_seconds / max(audio_seconds, 1e-8),
        "budgets": budgets,
        "oracle_wer": oracle.wer,
        "oracle_cer": oracle.cer,
    }
    write_json(args.output, result)
    if args.predictions:
        write_jsonl(args.predictions, rows)
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpeechMaster uncertainty-routed SSL-ASR cascade.")
    p.add_argument("--fast-model", default="facebook/wav2vec2-base-960h")
    p.add_argument("--strong-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--split", default="validation")
    p.add_argument("--subset", default="clean")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--route-percentages", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--predictions", default=None)
    return p


def main() -> None:
    print(evaluate_speechmaster(build_argparser().parse_args()))


if __name__ == "__main__":
    main()

