"""Train CTC head over cached features; evaluate WER/CER on dev each epoch."""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer

sys.path.insert(0, os.path.dirname(__file__))
from common import VOCAB_SIZE, BLANK, load_manifest, ids_to_text
from dataset import FeatDataset, collate_continuous, collate_discrete
from model import CTCHead


def greedy_decode(logits, lengths):
    """logits [B,T,V] -> list of decoded strings (CTC collapse)."""
    preds = logits.argmax(-1).cpu().numpy()  # [B,T]
    outs = []
    for b, L in enumerate(lengths.tolist()):
        seq = preds[b, :L]
        collapsed, prev = [], -1
        for p in seq:
            if p != prev and p != BLANK:
                collapsed.append(int(p))
            prev = p
        outs.append(ids_to_text(collapsed))
    return outs


def evaluate(model, loader, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for feats, flens, targets, tlens, uttids in loader:
            feats, flens = feats.to(device), flens.to(device)
            logits = model(feats, flens)
            hyps.extend(greedy_decode(logits, flens))
            # reconstruct refs from targets
            off = 0
            for L in tlens.tolist():
                refs.append(ids_to_text(targets[off:off + L].tolist()))
                off += L
    return wer(refs, hyps), cer(refs, hyps), refs, hyps


def load_discrete_labels(path):
    if path is None:
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--train_feat", required=True)
    ap.add_argument("--dev_feat", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--discrete_train", default=None, help="npz uttid->cluster ids")
    ap.add_argument("--discrete_dev", default=None)
    ap.add_argument("--discrete_vocab", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--exp", required=True, help="output dir")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.exp, exist_ok=True)
    device = args.device
    discrete = args.discrete_vocab is not None

    dtr = load_discrete_labels(args.discrete_train)
    dde = load_discrete_labels(args.discrete_dev)
    collate = collate_discrete if discrete else collate_continuous

    tr = FeatDataset(args.train_manifest, args.train_feat, args.layer, dtr)
    de = FeatDataset(args.dev_manifest, args.dev_feat, args.layer, dde)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=False)
    dl = DataLoader(de, batch_size=args.bs, shuffle=False, num_workers=4,
                    collate_fn=collate)

    model = CTCHead(VOCAB_SIZE, in_dim=768, hidden=args.hidden,
                    discrete_vocab=args.discrete_vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    n_params = sum(p.numel() for p in model.parameters())

    best = {"wer": 1e9}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for feats, flens, targets, tlens, _ in tl:
            feats, flens = feats.to(device), flens.to(device)
            targets, tlens = targets.to(device), tlens.to(device)
            logp = model(feats, flens).log_softmax(-1).transpose(0, 1)  # [T,B,V]
            loss = ctc(logp, targets, flens, tlens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        dw, dc, _, _ = evaluate(model, dl, device)
        print(f"[{args.tag}] ep{ep:02d} loss {tot/len(tl):.3f} dev_WER {dw*100:.2f} dev_CER {dc*100:.2f}")
        if dw < best["wer"]:
            best = {"wer": dw, "cer": dc, "epoch": ep}
            torch.save(model.state_dict(), os.path.join(args.exp, f"{args.tag}_best.pt"))

    dur = time.time() - t0
    result = {"tag": args.tag, "layer": args.layer, "discrete": discrete,
              "discrete_vocab": args.discrete_vocab, "params": n_params,
              "best_dev_wer": best["wer"], "best_dev_cer": best["cer"],
              "best_epoch": best["epoch"], "train_sec": dur}
    with open(os.path.join(args.exp, f"{args.tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.tag}] BEST dev_WER {best['wer']*100:.2f} (ep{best['epoch']}) "
          f"params {n_params/1e6:.2f}M time {dur:.0f}s")


if __name__ == "__main__":
    main()
