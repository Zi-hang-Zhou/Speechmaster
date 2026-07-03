"""Generate paper figures from experiment results.

Fig1: WER vs WavLM layer (ablation A, continuous).
Fig2: WER vs bitrate, continuous baseline vs discrete units (ablation B).
"""
import os
import sys
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.join(os.path.dirname(__file__), "..", "exp")
FIG = os.path.join(EXP, "..", "figures")


def load(pattern):
    out = []
    for f in glob.glob(os.path.join(EXP, pattern)):
        out.append(json.load(open(f)))
    return out


def fig_layer_sweep():
    rows = load("ablA_layers/layer*_result.json")
    rows = sorted(rows, key=lambda r: r["layer"])
    if not rows:
        print("no ablation A results")
        return
    xs = [r["layer"] for r in rows]
    wer = [r["best_dev_wer"] * 100 for r in rows]
    cer = [r["best_dev_cer"] * 100 for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.plot(xs, wer, "o-", label="WER")
    plt.plot(xs, cer, "s--", label="CER")
    plt.xlabel("WavLM hidden layer")
    plt.ylabel("Error rate (%)")
    plt.title("Dev error vs. representation layer (10h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_layer_sweep.png"), dpi=150)
    plt.close()
    print("fig1 saved; best layer =", min(rows, key=lambda r: r["best_dev_wer"])["layer"])


def fig_cont_vs_disc():
    # continuous point: best layer from ablation A
    contA = load("ablA_layers/layer*_result.json")
    if not contA:
        print("no continuous results")
        return
    best_cont = min(contA, key=lambda r: r["best_dev_wer"])
    disc = load("ablB_discrete/disc_k*_result.json")
    stats = {}
    for sf in glob.glob(os.path.join(EXP, "km", "k*_stats.json")):
        s = json.load(open(sf))
        stats[s["k"]] = s
    if not disc:
        print("no discrete results yet")
        return
    disc = sorted(disc, key=lambda r: r["discrete_vocab"])
    xs = [stats[r["discrete_vocab"]]["bitrate_bps"] for r in disc]
    wer = [r["best_dev_wer"] * 100 for r in disc]
    labels = [f"k={r['discrete_vocab']}" for r in disc]

    plt.figure(figsize=(5, 3.5))
    plt.plot(xs, wer, "o-", label="discrete units")
    for x, y, l in zip(xs, wer, labels):
        plt.annotate(l, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    # continuous baseline as horizontal line (768-dim fp16 ~ effectively continuous)
    plt.axhline(best_cont["best_dev_wer"] * 100, color="r", ls="--",
                label=f"continuous (L{best_cont['layer']})")
    plt.xlabel("Bitrate (bits/s)")
    plt.ylabel("Dev WER (%)")
    plt.title("Continuous vs. discrete units")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_cont_vs_disc.png"), dpi=150)
    plt.close()
    print("fig2 saved")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_layer_sweep()
    fig_cont_vs_disc()


if __name__ == "__main__":
    main()
