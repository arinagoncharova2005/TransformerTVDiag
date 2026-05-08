"""Sanity checks over Step C evidence bundles.

Reads:
  - logs/gaia/evidence/<config>/evidence_fold_{0..4}.jsonl
  - logs/gaia/attention/<config>/fold_{k}_index.csv  (for rcl_true/fti_true)

Produces:
  - logs/gaia/evidence/<config>/sanity_report.txt   (human-readable)
  - logs/gaia/evidence/<config>/sanity_per_sample.csv  (wide table)
  - logs/gaia/evidence/<config>/sanity_modality_top1.csv
  - logs/gaia/evidence/<config>/sanity_crosstab.csv

Run (from TransTVDiag_attention_analysis/):
    python experimental/evidence_sanity.py
    python experimental/evidence_sanity.py --config ce_awl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


MODALITIES = ("metric", "trace", "log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--evidence_root", default="logs/gaia/evidence")
    p.add_argument("--attn_root", default="logs/gaia/attention")
    return p.parse_args()


def has_any_raw(log_entries: list) -> bool:
    for entry in log_entries:
        for tpl in entry.get("details", {}).get("templates", []):
            if "raw_examples" in tpl and tpl["raw_examples"]:
                return True
    return False


def flatten_bundle(b: dict) -> dict:
    """Pull out per-modality top-1 fields into a flat row."""
    row = {
        "fold": b["fold"],
        "sample_idx": b["sample_idx"],
        "gaia_index": b["gaia_index"],
        "fti_pred": b["fti_pred"],
        "rcl_pred": b["rcl_pred"],
        "rcl_pred_service": b["rcl_pred_service"],
        "n_real_services": b["n_real_services"],
        "has_raw_log": has_any_raw(b["evidence_by_modality"].get("log", [])),
    }
    for m in MODALITIES:
        entries = b["evidence_by_modality"].get(m, [])
        if entries:
            top = entries[0]
            row[f"{m}_top1_service"] = top["service"]
            row[f"{m}_top1_service_idx"] = top["service_idx"]
            row[f"{m}_top1_weight"] = top["attention_weight"]
            row[f"{m}_top1_lift"] = top["attention_lift"]
            row[f"{m}_top1_is_rcl_pred"] = top["is_predicted_rcl"]
            row[f"{m}_top1_n_events"] = top["details"].get("n_events", 0)
        else:
            row[f"{m}_top1_service"] = None
            row[f"{m}_top1_service_idx"] = -1
            row[f"{m}_top1_weight"] = float("nan")
            row[f"{m}_top1_lift"] = float("nan")
            row[f"{m}_top1_is_rcl_pred"] = False
            row[f"{m}_top1_n_events"] = 0
    lifts = {m: row[f"{m}_top1_lift"] for m in MODALITIES}
    valid = {m: v for m, v in lifts.items() if not np.isnan(v)}
    row["winning_modality"] = (max(valid, key=valid.get)
                               if valid else None)
    return row


def load_evidence(config: str, folds: list, evidence_root: Path) -> pd.DataFrame:
    rows = []
    for fold in folds:
        path = evidence_root / config / f"evidence_fold_{fold}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                rows.append(flatten_bundle(json.loads(line)))
    return pd.DataFrame(rows)


def load_truth(config: str, folds: list, attn_root: Path) -> pd.DataFrame:
    """Pull rcl_true / fti_true from each fold index csv."""
    out = []
    for fold in folds:
        path = attn_root / config / f"fold_{fold}_index.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        out.append(df[["fold", "sample_idx", "gaia_index",
                       "rcl_true", "rcl_pred", "fti_true", "fti_pred"]])
    return pd.concat(out, ignore_index=True)


def fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num}/{denom} ({100 * num / denom:.1f}%)"


def write_report(df: pd.DataFrame, out_path: Path,
                 config: str, modality_top1: pd.DataFrame,
                 crosstab: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Evidence sanity report — config={config}")
    lines.append(f"Total bundles: {len(df)}")
    lines.append("")

    lines.append("## 1. Coverage")
    lines.append(f"With at least one raw log example: "
                 f"{fmt_pct(int(df['has_raw_log'].sum()), len(df))}")
    for m in MODALITIES:
        nonzero = int((df[f'{m}_top1_n_events'] > 0).sum())
        lines.append(f"top-1 {m} has events: {fmt_pct(nonzero, len(df))}")
    lines.append("")

    lines.append("## 2. Winning modality (argmax of top-1 lift over modalities)")
    win_counts = df["winning_modality"].value_counts(dropna=False)
    for mod, cnt in win_counts.items():
        lines.append(f"  {mod}: {fmt_pct(int(cnt), len(df))}")
    lines.append("")

    lines.append("## 3. Per-modality top-1 lift (median / mean / p90)")
    for m in MODALITIES:
        col = df[f"{m}_top1_lift"].dropna()
        lines.append(f"  {m}: median={col.median():.3f}, mean={col.mean():.3f}, "
                     f"p90={col.quantile(0.9):.3f}, p99={col.quantile(0.99):.3f}")
    lines.append("")

    lines.append("## 4. Attention top-1 service == rcl_pred (per modality)")
    for m in MODALITIES:
        match = int(df[f"{m}_top1_is_rcl_pred"].sum())
        lines.append(f"  {m}: {fmt_pct(match, len(df))}")
    lines.append("")

    lines.append("## 5. Model accuracy (from fold index csv)")
    if "rcl_correct" in df.columns:
        lines.append(f"  RCL accuracy: "
                     f"{fmt_pct(int(df['rcl_correct'].sum()), len(df))}")
    if "fti_correct" in df.columns:
        lines.append(f"  FTI accuracy: "
                     f"{fmt_pct(int(df['fti_correct'].sum()), len(df))}")
    lines.append("")

    lines.append("## 6. Cross-tab: attention top-1 matches rcl_pred  ×  RCL correct")
    lines.append("(rows: attention–prediction agreement; cols: model RCL correct)")
    lines.append("")
    lines.append(crosstab.to_string())
    lines.append("")

    lines.append("## 7. Lift conditional on RCL correctness")
    lines.append("Median top-1 lift, split by rcl_correct:")
    if "rcl_correct" in df.columns:
        for m in MODALITIES:
            grp = df.groupby("rcl_correct")[f"{m}_top1_lift"].median()
            lines.append(f"  {m}: " + ", ".join(
                f"{'correct' if k else 'incorrect'}={v:.3f}"
                for k, v in grp.items()))
    lines.append("")

    lines.append("## 8. Per-fold breakdown")
    grp = df.groupby("fold").agg(
        n=("gaia_index", "size"),
        rcl_acc=("rcl_correct", "mean") if "rcl_correct" in df.columns else ("gaia_index", "size"),
        fti_acc=("fti_correct", "mean") if "fti_correct" in df.columns else ("gaia_index", "size"),
        with_raw=("has_raw_log", "mean"),
    )
    lines.append(grp.to_string())
    lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    evidence_root = Path(args.evidence_root)
    attn_root = Path(args.attn_root)
    out_dir = evidence_root / args.config

    print(f"loading evidence bundles from {out_dir}/")
    df = load_evidence(args.config, folds, evidence_root)
    print(f"  {len(df)} bundles")

    print(f"loading truth labels from {attn_root}/{args.config}/fold_*_index.csv")
    truth = load_truth(args.config, folds, attn_root)
    print(f"  {len(truth)} truth rows")

    df = df.merge(truth[["fold", "sample_idx", "rcl_true", "fti_true"]],
                  on=["fold", "sample_idx"], how="left")
    df["rcl_correct"] = df["rcl_pred"] == df["rcl_true"]
    fti_pred_num = pd.to_numeric(df["fti_pred"], errors="coerce")
    fti_true_num = pd.to_numeric(df["fti_true"], errors="coerce")
    df["fti_correct"] = (fti_pred_num == fti_true_num) & fti_pred_num.notna()

    print()
    print("== summary ==")
    print(f"bundles: {len(df)}")
    print(f"RCL correct: {df['rcl_correct'].mean():.4f}")
    print(f"FTI correct: {df['fti_correct'].mean():.4f}")
    print(f"winning modality:")
    print(df["winning_modality"].value_counts(dropna=False).to_string())

    modality_top1 = pd.DataFrame({
        "modality": list(MODALITIES),
        "wins": [int((df["winning_modality"] == m).sum()) for m in MODALITIES],
        "median_top1_lift": [df[f"{m}_top1_lift"].median() for m in MODALITIES],
        "mean_top1_lift": [df[f"{m}_top1_lift"].mean() for m in MODALITIES],
        "match_rcl_pred_pct": [
            100 * df[f"{m}_top1_is_rcl_pred"].mean() for m in MODALITIES],
    })

    rows = []
    for m in MODALITIES:
        ct = pd.crosstab(df[f"{m}_top1_is_rcl_pred"], df["rcl_correct"],
                         margins=True, margins_name="all")
        ct.index = [f"{m}: top1==rcl_pred={i}" for i in ct.index]
        rows.append(ct)
    crosstab = pd.concat(rows)

    df.to_csv(out_dir / "sanity_per_sample.csv", index=False)
    modality_top1.to_csv(out_dir / "sanity_modality_top1.csv", index=False)
    crosstab.to_csv(out_dir / "sanity_crosstab.csv")

    report_path = out_dir / "sanity_report.txt"
    write_report(df, report_path, args.config, modality_top1, crosstab)

    print()
    print(f"wrote: {out_dir}/sanity_per_sample.csv")
    print(f"wrote: {out_dir}/sanity_modality_top1.csv")
    print(f"wrote: {out_dir}/sanity_crosstab.csv")
    print(f"wrote: {report_path}")
    print()
    print("--- report preview ---")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
