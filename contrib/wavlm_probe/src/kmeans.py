"""K-means discretization of cached WavLM features -> discrete unit labels.

Fit k-means on a sample of train frames, then assign cluster ids to every
utterance in the given manifests. Also compute token-rate / bitrate stats.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(__file__))
from common import load_manifest


def load_feats(manifest, feat_dir, layer):
    key = f"L{layer}"
    items = load_manifest(manifest)
    for it in items:
        arr = np.load(os.path.join(feat_dir, it["uttid"] + ".npz"))[key]
        yield it["uttid"], arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--train_feat", required=True)
    ap.add_argument("--assign_manifests", nargs="+", required=True)
    ap.add_argument("--assign_feats", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--sample_frames", type=int, default=300000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # collect a frame sample to fit k-means
    pool, n = [], 0
    for _, arr in tqdm(load_feats(args.train_manifest, args.train_feat, args.layer),
                       desc="collect frames"):
        pool.append(arr)
        n += arr.shape[0]
        if n >= args.sample_frames:
            break
    X = np.concatenate(pool, 0)[: args.sample_frames]
    print(f"fit k-means k={args.k} on {X.shape[0]} frames")
    km = MiniBatchKMeans(n_clusters=args.k, batch_size=10000, max_iter=100,
                         n_init=3, random_state=42, verbose=0)
    km.fit(X)

    # assign every utterance in each manifest
    frame_rate = 50.0  # WavLM base frame rate (Hz)
    for man, fdir in zip(args.assign_manifests, args.assign_feats):
        labels = {}
        for uttid, arr in tqdm(load_feats(man, fdir, args.layer),
                               desc=f"assign {os.path.basename(man)}"):
            labels[uttid] = km.predict(arr).astype(np.int16)
        name = os.path.basename(man).replace(".jsonl", "")
        np.savez(os.path.join(args.out_dir, f"{args.tag}_{name}.npz"), **labels)

    # bitrate stats (no dedup): tokens/sec and bits/sec
    token_rate = frame_rate
    bitrate = token_rate * np.log2(args.k)
    stats = {"tag": args.tag, "layer": args.layer, "k": args.k,
             "token_rate_hz": token_rate, "bitrate_bps": bitrate,
             "bits_per_token": float(np.log2(args.k))}
    with open(os.path.join(args.out_dir, f"{args.tag}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"k={args.k} token_rate {token_rate}Hz bitrate {bitrate:.0f} bps")


if __name__ == "__main__":
    main()
