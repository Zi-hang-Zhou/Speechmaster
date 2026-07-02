from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_jsons(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return rows


def summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr_rows = []
    if args.asr:
        for row in _load_jsons(args.asr):
            rep = "final"
            if row.get("blend"):
                rep = row["blend"]
            elif row.get("layer") is not None:
                rep = f"layer {row['layer']}"
            asr_rows.append(
                {
                    "System": row["model"].split("/")[-1],
                    "Rep.": rep,
                    "N": row["limit"],
                    "WER": 100 * row["wer"],
                    "CER": 100 * row["cer"],
                    "RTF": row["rtf"],
                }
            )
        asr_df = pd.DataFrame(asr_rows)
        asr_df.to_csv(out_dir / "asr_metrics.csv", index=False)
        (out_dir / "asr_metrics.tex").write_text(
            asr_df.to_latex(index=False, float_format="%.3f"),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.2, 3.2))
        plt.bar(asr_df["System"] + "/" + asr_df["Rep."].astype(str), asr_df["WER"])
        plt.ylabel("WER (%)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "asr_wer.pdf")
        plt.close()

    unit_rows = []
    if args.units:
        for row in _load_jsons(args.units):
            model = row["model"].split("/")[-1]
            for item in row["unit_results"]:
                unit_rows.append(
                    {
                        "System": model,
                        "K": item["codebook_size"],
                        "Tok/s": item["token_rate"],
                        "Dedup tok/s": item["dedup_token_rate"],
                        "Bit/s": item["bitrate"],
                        "Dedup bit/s": item["dedup_bitrate"],
                    }
                )
        unit_df = pd.DataFrame(unit_rows)
        unit_df.to_csv(out_dir / "unit_metrics.csv", index=False)
        (out_dir / "unit_metrics.tex").write_text(
            unit_df.to_latex(index=False, float_format="%.2f"),
            encoding="utf-8",
        )
        plt.figure(figsize=(5.8, 3.2))
        for system, group in unit_df.groupby("System"):
            plt.plot(group["K"], group["Dedup bit/s"], marker="o", label=system)
        plt.xlabel("Codebook size")
        plt.ylabel("Deduplicated bitrate (bit/s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "unit_bitrate.pdf")
        plt.close()

    if args.speechmaster:
        rows = []
        for row in _load_jsons(args.speechmaster):
            for item in row["budgets"]:
                rows.append(
                    {
                        "Route": f"{item['route_percentage']:.0f}\\%",
                        "Routed utt.": item["routed_utterances"],
                        "WER": 100 * item["wer"],
                        "CER": 100 * item["cer"],
                        "RTF": item["rtf"],
                    }
                )
            rows.append(
                {
                    "Route": "Oracle",
                    "Routed utt.": "-",
                    "WER": 100 * row["oracle_wer"],
                    "CER": 100 * row["oracle_cer"],
                    "RTF": "-",
                }
            )
        sm_df = pd.DataFrame(rows)
        sm_df.to_csv(out_dir / "speechmaster_metrics.csv", index=False)
        (out_dir / "speechmaster_metrics.tex").write_text(
            sm_df.to_latex(index=False, float_format="%.3f"),
            encoding="utf-8",
        )
        numeric = sm_df[sm_df["Route"].astype(str).str.endswith("\\%")].copy()
        numeric["RouteNum"] = numeric["Route"].str.replace("\\%", "", regex=False).astype(float)
        plt.figure(figsize=(5.8, 3.2))
        plt.plot(numeric["RouteNum"], numeric["WER"], marker="o", label="WER")
        plt.xlabel("Routed utterances (%)")
        plt.ylabel("WER (%)")
        plt.tight_layout()
        plt.savefig(out_dir / "speechmaster_budget.pdf")
        plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize experiment JSON files into paper assets.")
    p.add_argument("--asr", nargs="*", default=[])
    p.add_argument("--units", nargs="*", default=[])
    p.add_argument("--speechmaster", nargs="*", default=[])
    p.add_argument("--output-dir", default="results/tables")
    return p


def main() -> None:
    summarize(build_argparser().parse_args())


if __name__ == "__main__":
    main()
