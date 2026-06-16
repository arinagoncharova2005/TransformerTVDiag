"""Step E.4 — assemble the explanation-evaluation results into one report.

Reads the CSVs produced by eval_faithfulness.py, eval_utility.py and
eval_llm_judge.py from logs/gaia/explanation_eval/<config>/ and writes a single
markdown summary plus a few figures. Each section is skipped gracefully if its
input CSV is missing, so the report works whether or not the judge has been run.

Writes:
  logs/gaia/explanation_eval/<config>/REPORT.md
  logs/gaia/explanation_eval/<config>/fig_faithfulness_by_group.png
  logs/gaia/explanation_eval/<config>/fig_judge_vs_halluc.png   (if judge present)

Run (from TransTVDiag_attention_analysis/):
    python experimental/eval_report.py --config cb_ce_awl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--eval_root", default="logs/gaia/explanation_eval")
    return p.parse_args()


def read_txt(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def main() -> None:
    args = parse_args()
    d = Path(args.eval_root) / args.config
    if not d.exists():
        raise SystemExit(f"{d} not found — run the eval scripts first")

    md = [f"# Explanation evaluation — `{args.config}`", ""]

    # ----- 1. faithfulness -----
    faith_csv = d / "faithfulness_per_sample.csv"
    md.append("## 1. Faithfulness (objective, hallucination check)")
    if faith_csv.exists():
        f = pd.read_csv(faith_csv)
        f["halluc_rate"] = f["bad_mentions"] / f["total_mentions"].clip(lower=1)
        md.append(f"- samples: **{len(f)}**")
        md.append(f"- atom-level hallucination rate (service/error names): "
                  f"**{f['bad_mentions'].sum() / max(f['total_mentions'].sum(), 1):.4f}**")
        md.append(f"- fully grounded (atom-level): **{f['fully_grounded'].mean():.3f}**")
        md.append(f"- explanation names the predicted root cause: "
                  f"**{f['mentions_pred_rcl'].mean():.3f}**")
        if "relational_claims" in f:
            md.append(f"- relational hallucination rate (wrong peer/direction/binding): "
                      f"**{f['relational_bad'].sum() / max(f['relational_claims'].sum(), 1):.4f}** "
                      f"({int(f['relational_bad'].sum())}/{int(f['relational_claims'].sum())} edges)")
            md.append(f"- modality misattribution rate: "
                      f"**{f['modality_bad'].sum() / max(f['modality_claims'].sum(), 1):.4f}** "
                      f"({int(f['modality_bad'].sum())}/{int(f['modality_claims'].sum())} attributions)")
            md.append(f"- **fully grounded (strict: atom+relational+modality): "
                      f"{f['fully_grounded_strict'].mean():.3f}**")
        md.append("")
        g = f.groupby("group").agg(
            n=("gaia_index", "size"),
            halluc_rate=("halluc_rate", "mean"),
            grounded=("fully_grounded", "mean")).reset_index()
        md.append(g.to_markdown(index=False, floatfmt=".4f"))
        md.append("")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(g["group"], g["grounded"], color="#4c72b0")
        ax.set_ylabel("fully grounded rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Faithfulness by group")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(d / "fig_faithfulness_by_group.png", dpi=130)
        plt.close(fig)
        md.append("![faithfulness](fig_faithfulness_by_group.png)")
    else:
        md.append("_faithfulness_per_sample.csv not found — run eval_faithfulness.py_")
    md.append("")

    # ----- 2. utility -----
    md.append("## 2. Utility (error detection + corrective recovery)")
    util_txt = read_txt(d / "utility_report.txt")
    if util_txt:
        md.append("```")
        md.append(util_txt.strip())
        md.append("```")
    else:
        md.append("_utility_report.txt not found — run eval_utility.py_")
    md.append("")

    # ----- 3. LLM judge -----
    md.append("## 3. LLM-as-judge (subjective, model-graded)")
    judge_csv = d / "judge_per_sample.csv"
    if judge_csv.exists():
        j = pd.read_csv(judge_csv)
        md.append(f"- samples judged: **{len(j)}**")
        for col in ("judge_faithfulness", "judge_completeness", "judge_actionability"):
            s = j[col].dropna()
            if len(s):
                md.append(f"- {col}: mean **{s.mean():.2f}** (median {s.median():.0f})")
        md.append("")
        md.append("```")
        md.append(read_txt(d / "judge_report.txt").strip())
        md.append("```")

        # validation scatter vs objective faithfulness
        if faith_csv.exists():
            f = pd.read_csv(faith_csv)
            f["halluc_rate"] = f["bad_mentions"] / f["total_mentions"].clip(lower=1)
            m = j.merge(f[["gaia_index", "halluc_rate"]], on="gaia_index").dropna(
                subset=["judge_faithfulness"])
            if len(m) >= 3:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(m["halluc_rate"], m["judge_faithfulness"],
                           alpha=0.6, color="#c44e52")
                ax.set_xlabel("objective hallucination rate")
                ax.set_ylabel("judge faithfulness (1-5)")
                ax.set_title("Judge vs objective faithfulness")
                plt.tight_layout()
                fig.savefig(d / "fig_judge_vs_halluc.png", dpi=130)
                plt.close(fig)
                md.append("")
                md.append("![judge vs halluc](fig_judge_vs_halluc.png)")
    else:
        md.append("_judge_per_sample.csv not found — run eval_llm_judge.py (needs LLM API)_")
    md.append("")

    md.append("## Limitations")
    md.append("- Single config (`{}`), N≈200 explanations, one dataset (GAIA).".format(
        args.config))
    md.append("- Faithfulness is *evidence-bounded*: it checks consistency with the "
              "given evidence, not absolute correctness of the diagnosis.")
    md.append("- Judge and explainer may share model-family biases; judge scores are "
              "indicative, not ground truth.")
    md.append("- `contradict` is a signal, not a calibrated confidence — see the "
              "error-detection precision/recall above.")

    out_path = d / "REPORT.md"
    out_path.write_text("\n".join(md) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
