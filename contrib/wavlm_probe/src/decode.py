"""Decode a trained CTC head on a test manifest; report WER/CER + error breakdown."""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from jiwer import process_words

sys.path.insert(0, os.path.dirname(__file__))
from common import VOCAB_SIZE, load_manifest
from dataset import FeatDataset, collate_continuous, collate_discrete
from model import CTCHead
from train import evaluate, load_discrete_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--test_feat", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--discrete_test", default=None)
    ap.add_argument("--discrete_vocab", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", required=True, help="output json for results")
    ap.add_argument("--dump_hyp", default=None, help="optional txt of ref/hyp pairs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    discrete = args.discrete_vocab is not None
    collate = collate_discrete if discrete else collate_continuous
    dtest = load_discrete_labels(args.discrete_test)

    ds = FeatDataset(args.test_manifest, args.test_feat, args.layer, dtest)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    model = CTCHead(VOCAB_SIZE, in_dim=768, hidden=args.hidden,
                    discrete_vocab=args.discrete_vocab).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    w, c, refs, hyps = evaluate(model, dl, device)
    out = process_words(refs, hyps)
    total = out.substitutions + out.deletions + out.insertions + out.hits
    result = {
        "ckpt": args.ckpt, "layer": args.layer,
        "discrete_vocab": args.discrete_vocab,
        "test_wer": w, "test_cer": c,
        "substitutions": out.substitutions, "deletions": out.deletions,
        "insertions": out.insertions, "hits": out.hits,
        "n_ref_words": total,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if args.dump_hyp:
        with open(args.dump_hyp, "w") as f:
            for r, h in zip(refs, hyps):
                f.write(f"REF: {r}\nHYP: {h}\n\n")
    print(f"test_WER {w*100:.2f} test_CER {c*100:.2f} "
          f"| S {out.substitutions} D {out.deletions} I {out.insertions}")


if __name__ == "__main__":
    main()
