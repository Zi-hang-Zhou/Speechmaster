from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from .data import duration_seconds, load_librispeech_split
from .io import write_json


def _encoder_model_id(model_id: str) -> str:
    # Most CTC checkpoints keep the same processor and expose an encoder through
    # AutoModel. If the checkpoint does not, users can pass the base SSL model.
    return model_id


def extract_hidden(model, processor, sample: dict, layer: int, device: torch.device) -> np.ndarray:
    inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.inference_mode():
        outputs = model(input_values, output_hidden_states=True)
    hs = outputs.hidden_states[layer].squeeze(0).detach().cpu().float().numpy()
    return hs


def deduplicate(tokens: np.ndarray) -> np.ndarray:
    if len(tokens) == 0:
        return tokens
    keep = np.ones(len(tokens), dtype=bool)
    keep[1:] = tokens[1:] != tokens[:-1]
    return tokens[keep]


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(_encoder_model_id(args.model)).to(device)
    model.eval()

    samples = load_librispeech_split(
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        streaming=not args.no_streaming,
        cache_dir=args.cache_dir,
    )

    train_frames = []
    rng = np.random.default_rng(args.seed)
    total_audio = sum(duration_seconds(s) for s in samples)
    start = time.perf_counter()
    for sample in tqdm(samples, desc="extract"):
        hs = extract_hidden(model, processor, sample, args.layer, device)
        if len(hs) > args.frames_per_utt:
            idx = rng.choice(len(hs), size=args.frames_per_utt, replace=False)
            hs = hs[idx]
        train_frames.append(hs)
    frame_matrix = np.concatenate(train_frames, axis=0)
    if len(frame_matrix) > args.max_frames:
        idx = rng.choice(len(frame_matrix), size=args.max_frames, replace=False)
        frame_matrix = frame_matrix[idx]

    results = []
    for k in args.codebooks:
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(4096, max(256, len(frame_matrix))),
            random_state=args.seed,
            n_init="auto",
            max_iter=args.max_iter,
        )
        km.fit(frame_matrix)
        token_count = 0
        dedup_count = 0
        inertia = 0.0
        for sample in tqdm(samples, desc=f"k={k}"):
            hs = extract_hidden(model, processor, sample, args.layer, device)
            tokens = km.predict(hs)
            token_count += len(tokens)
            dedup_count += len(deduplicate(tokens))
            inertia += float(km.score(hs))
        token_rate = token_count / max(total_audio, 1e-8)
        dedup_token_rate = dedup_count / max(total_audio, 1e-8)
        bitrate = token_rate * math.log2(k)
        dedup_bitrate = dedup_token_rate * math.log2(k)
        results.append(
            {
                "codebook_size": k,
                "token_rate": token_rate,
                "dedup_token_rate": dedup_token_rate,
                "bitrate": bitrate,
                "dedup_bitrate": dedup_bitrate,
                "mean_negative_inertia": -inertia / max(token_count, 1),
            }
        )

    out = {
        "model": args.model,
        "split": args.split,
        "subset": args.subset,
        "limit": len(samples),
        "layer": args.layer,
        "audio_seconds": total_audio,
        "elapsed_seconds": time.perf_counter() - start,
        "device": str(device),
        "unit_results": results,
    }
    write_json(args.output, out)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze discrete SSL units with k-means.")
    p.add_argument("--model", default="facebook/hubert-base-ls960")
    p.add_argument("--split", default="validation")
    p.add_argument("--subset", default="clean")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--codebooks", type=int, nargs="+", default=[50, 100, 200, 500])
    p.add_argument("--frames-per-utt", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    print(analyze(args))


if __name__ == "__main__":
    main()
