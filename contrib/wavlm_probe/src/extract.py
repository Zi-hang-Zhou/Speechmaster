"""Extract & cache frozen WavLM hidden states for all utterances in a manifest.

For each utterance we run WavLM once and save the requested layers' hidden
states as fp16 .npy, so downstream training (layer sweep, discretization) reads
from disk without repeated forward passes.
"""
import os
import sys
import argparse

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.dirname(__file__))
from common import load_manifest

DEFAULT_MODEL = "microsoft/wavlm-base-plus"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output feature dir")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[1, 4, 6, 8, 10, 12],
                    help="hidden_states indices to cache (0=embeddings)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_flush", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = WavLMModel.from_pretrained(args.model).to(dev).eval().half()

    items = load_manifest(args.manifest)
    max_layer = max(args.layers)
    for it in tqdm(items, desc=f"extract {os.path.basename(args.manifest)}"):
        out_path = os.path.join(args.out, it["uttid"] + ".npz")
        if os.path.exists(out_path):
            continue
        wav, sr = sf.read(it["flac"])
        assert sr == 16000
        iv = fe(wav, sampling_rate=16000, return_tensors="pt").input_values
        iv = iv.to(dev).half()
        hs = model(iv, output_hidden_states=True).hidden_states
        feats = {f"L{l}": hs[l][0].float().cpu().numpy().astype(np.float16)
                 for l in args.layers}
        np.savez(out_path, **feats)

    print(f"done: {len(items)} utts -> {args.out} (layers {args.layers})")


if __name__ == "__main__":
    main()
