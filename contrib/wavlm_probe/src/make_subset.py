"""Build low-resource subset manifests (10h / 1h) from train-clean-100.

Sample utterances speaker-balanced until target duration is reached, with a
fixed seed for reproducibility. Also emit full dev/test manifests.
"""
import os
import sys
import json
import random
import argparse
from collections import defaultdict

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import parse_librispeech, save_manifest

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "LibriSpeech")


def with_durations(items):
    for it in tqdm(items, desc="probe dur"):
        info = sf.info(it["flac"])
        it["dur"] = info.frames / info.samplerate
    return items


def sample_subset(items, target_hours, seed):
    """Round-robin over speakers, picking random utterances until target."""
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for it in items:
        spk = it["uttid"].split("-")[0]
        by_spk[spk].append(it)
    for spk in by_spk:
        rng.shuffle(by_spk[spk])
    spks = sorted(by_spk.keys())
    rng.shuffle(spks)

    target = target_hours * 3600.0
    chosen, total = [], 0.0
    ptr = {s: 0 for s in spks}
    while total < target:
        progressed = False
        for s in spks:
            if ptr[s] < len(by_spk[s]):
                it = by_spk[s][ptr[s]]
                ptr[s] += 1
                chosen.append(it)
                total += it["dur"]
                progressed = True
                if total >= target:
                    break
        if not progressed:
            break  # exhausted all utterances
    chosen.sort(key=lambda x: x["uttid"])
    return chosen, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="LibriSpeech root containing train/dev/test splits")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # dev / test: full splits
    for split in ["dev-clean", "test-clean"]:
        items = parse_librispeech(os.path.join(args.root, split))
        items = with_durations(items)
        save_manifest(items, os.path.join(args.out, f"{split}.jsonl"))
        print(f"{split}: {len(items)} utts, {sum(i['dur'] for i in items)/3600:.2f} h")

    # train subsets from train-clean-100
    train = parse_librispeech(os.path.join(args.root, "train-clean-100"))
    train = with_durations(train)
    save_manifest(train, os.path.join(args.out, "train-clean-100.jsonl"))
    print(f"train-clean-100 full: {len(train)} utts, {sum(i['dur'] for i in train)/3600:.2f} h")

    for hours in [10, 1]:
        sub, total = sample_subset(train, hours, args.seed)
        path = os.path.join(args.out, f"train-{hours}h.jsonl")
        save_manifest(sub, path)
        nspk = len({i["uttid"].split("-")[0] for i in sub})
        print(f"train-{hours}h: {len(sub)} utts, {total/3600:.2f} h, {nspk} speakers -> {path}")


if __name__ == "__main__":
    main()
