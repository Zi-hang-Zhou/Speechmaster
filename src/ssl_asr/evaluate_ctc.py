from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor

from .data import duration_seconds, load_librispeech_split
from .io import write_json, write_jsonl
from .metrics import compute_asr_metrics


def _decode_logits(processor, logits: torch.Tensor) -> str:
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def _logits_from_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "dropout"):
        hidden = model.dropout(hidden)
    return model.lm_head(hidden)


def _select_logits(model, outputs, layer: int | None, blend: str | None) -> torch.Tensor:
    if layer is None and blend is None:
        return outputs.logits
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("Model did not return hidden states.")
    if layer is not None:
        return _logits_from_hidden(model, hidden_states[layer])
    if blend == "last4":
        hs = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
    elif blend == "midlast":
        mid = len(hidden_states) // 2
        hs = 0.5 * hidden_states[mid] + 0.5 * hidden_states[-1]
    else:
        raise ValueError(f"Unsupported blend: {blend}")
    return _logits_from_hidden(model, hs)


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device)
    model.eval()

    samples = load_librispeech_split(
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        streaming=not args.no_streaming,
        cache_dir=args.cache_dir,
    )

    rows = []
    references, hypotheses = [], []
    decode_time = 0.0
    audio_time = 0.0
    with torch.inference_mode():
        for sample in tqdm(samples, desc=Path(args.model).name):
            audio_time += duration_seconds(sample)
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
            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=args.layer is not None or args.blend is not None,
            )
            logits = _select_logits(model, outputs, args.layer, args.blend)
            hyp = _decode_logits(processor, logits.cpu())
            decode_time += time.perf_counter() - start
            references.append(sample["text"])
            hypotheses.append(hyp)
            rows.append(
                {
                    "id": sample["id"],
                    "reference": sample["text"],
                    "hypothesis": hyp,
                    "duration": duration_seconds(sample),
                }
            )

    metrics = compute_asr_metrics(references, hypotheses)
    result = {
        "model": args.model,
        "split": args.split,
        "subset": args.subset,
        "limit": len(samples),
        "layer": args.layer,
        "blend": args.blend,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "substitutions": metrics.substitutions,
        "deletions": metrics.deletions,
        "insertions": metrics.insertions,
        "hits": metrics.hits,
        "audio_seconds": audio_time,
        "decode_seconds": decode_time,
        "rtf": decode_time / max(audio_time, 1e-8),
        "device": str(device),
    }
    write_json(args.output, result)
    if args.predictions:
        write_jsonl(args.predictions, rows)
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a pretrained CTC SSL-ASR model.")
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--subset", default="clean")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--layer", type=int, default=None, help="Decode with an intermediate hidden layer.")
    p.add_argument("--blend", choices=["last4", "midlast"], default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions", default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    result = evaluate(args)
    print(result)


if __name__ == "__main__":
    main()

