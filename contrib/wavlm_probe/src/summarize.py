"""Collect all experiment result JSONs into results.csv."""
import os
import sys
import csv
import glob
import json

EXP = os.path.join(os.path.dirname(__file__), "..", "exp")


def main():
    rows = []
    for jf in sorted(glob.glob(os.path.join(EXP, "**", "*result*.json"), recursive=True)):
        with open(jf) as f:
            d = json.load(f)
        d["_file"] = os.path.relpath(jf, EXP)
        rows.append(d)
    if not rows:
        print("no results found")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = os.path.join(EXP, "..", "results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out}")
    for r in rows:
        tag = r.get("tag", r.get("_file"))
        wer = r.get("best_dev_wer", r.get("test_wer"))
        print(f"  {tag:24s} WER={wer*100:.2f}" if wer is not None else f"  {tag}")


if __name__ == "__main__":
    main()
