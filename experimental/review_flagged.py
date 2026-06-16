"""Dump the explanations flagged by eval_faithfulness for manual review.

For each sample with a relational or modality flag, prints: the flagged item,
the explanation text, and the TRACE section of the prompt it was judged against.
Read each one and decide whether it is a real hallucination or a parser false
positive, then correct the relational/modality rates accordingly.

Run (where the sample texts live, e.g. the server):
    python experimental/review_flagged.py --config cb_ce_awl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--explanations_root", default="logs/gaia/explanations")
    p.add_argument("--eval_root", default="logs/gaia/explanation_eval")
    return p.parse_args()


def trace_section(prompt: str) -> str:
    start = prompt.find("--- TRACE")
    end = prompt.find("--- LOG")
    return prompt[start:end].rstrip() if start >= 0 else "(no trace section)"


def modality_sections(prompt: str) -> str:
    """Return the LOG and METRIC sections, needed to check modality flags."""
    start = prompt.find("--- LOG")
    if start < 0:
        return "(no log/metric section)"
    return prompt[start:].rstrip()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_root) / args.config
    expl_dir = Path(args.explanations_root) / args.config

    f = pd.read_csv(eval_dir / "faithfulness_per_sample.csv")
    flagged = f[(f["relational_bad"] > 0) | (f["modality_bad"] > 0)]
    print(f"flagged samples: {len(flagged)} "
          f"(relational edges: {int(f['relational_bad'].sum())}, "
          f"modality: {int(f['modality_bad'].sum())})\n")

    for _, row in flagged.iterrows():
        gi = int(row["gaia_index"])
        with open(expl_dir / f"sample_{gi}.json") as fh:
            s = json.load(fh)
        p = s.get("parsed", {})
        print("=" * 78)
        print(f"gaia={gi}  group={row['group']}  rcl_pred={s.get('rcl_pred_service')}")
        if row["relational_bad"] > 0:
            print(f"  FLAGGED EDGES: {row['bad_edges']}")
        if row["modality_bad"] > 0:
            print(f"  FLAGGED MODALITY: {row['bad_modality']}")
        print(f"\n  evidence_summary: {p.get('evidence_summary')}")
        print(f"  explanation:      {p.get('explanation')}")
        print(f"\n  {trace_section(s.get('prompt', ''))}")
        if row["modality_bad"] > 0:  # log/metric sections needed to judge these
            print(f"\n  {modality_sections(s.get('prompt', ''))}")
        print()


if __name__ == "__main__":
    main()
