"""Step E.2 — utility of LLM explanations: error detection + corrective recovery.

Two objective evaluations against ground truth (no LLM calls):

(a) Error detection (uses summary.csv -> all generated verdicts, no text needed):
    Can the `rcl_agreement` verdict flag cases where the model's RCL is wrong?
    Treat "contradict" (and, separately, "contradict|unclear") as predicting
    "model is wrong" and score precision/recall/F1 vs the always-trust baseline.

(b) Corrective recovery (uses sample_*.json text + sanity_per_sample.csv truth):
    When the model is wrong and the LLM says "contradict", does the explanation
    actually mention the *true* root-cause service? That would mean explanations
    do not just describe but can redirect the diagnosis.

Ground-truth root cause = `rcl_true` (label index) in sanity_per_sample.csv,
mapped to a service name via the (service_idx -> service) table reconstructed
from the evidence bundles.

Writes:
  logs/gaia/explanation_eval/<config>/utility_per_sample.csv
  logs/gaia/explanation_eval/<config>/utility_report.txt

Run (from TransTVDiag_attention_analysis/):
    python experimental/eval_utility.py --config cb_ce_awl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from eval_faithfulness import extract_services


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl")
    p.add_argument("--explanations_root", default="logs/gaia/explanations")
    p.add_argument("--evidence_root", default="logs/gaia/evidence")
    p.add_argument("--out_root", default="logs/gaia/explanation_eval")
    return p.parse_args()


# ---------- helpers ---------------------------------------------------------

def build_idx_to_service(evidence_dir: Path) -> dict[int, str]:
    """Reconstruct label-index -> service-name map from evidence bundles."""
    mapping: dict[int, str] = {}
    for jsonl_path in sorted(evidence_dir.glob("evidence_fold_*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                bundle = json.loads(line)
                for entries in bundle["evidence_by_modality"].values():
                    for e in entries:
                        mapping[int(e["service_idx"])] = e["service"]
        if len(mapping) >= 10:
            break
    return mapping


def confusion(flag_pred: list[bool], is_wrong: list[bool]) -> dict:
    tp = sum(p and w for p, w in zip(flag_pred, is_wrong))
    fp = sum(p and not w for p, w in zip(flag_pred, is_wrong))
    fn = sum(not p and w for p, w in zip(flag_pred, is_wrong))
    tn = sum(not p and not w for p, w in zip(flag_pred, is_wrong))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision,
                recall=recall, f1=f1, specificity=specificity)


# ---------- main ------------------------------------------------------------

def main() -> None:
    args = parse_args()
    expl_dir = Path(args.explanations_root) / args.config
    evidence_dir = Path(args.evidence_root) / args.config
    out_dir = Path(args.out_root) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(expl_dir / "summary.csv")
    summary = summary[summary["status"] == "ok"].copy()
    summary["rcl_correct"] = summary["rcl_correct"].astype(bool)
    summary["is_wrong"] = ~summary["rcl_correct"]

    sanity = pd.read_csv(evidence_dir / "sanity_per_sample.csv")
    truth = sanity.set_index("gaia_index")[["rcl_true", "fti_true"]]
    summary = summary.join(truth, on="gaia_index")

    lines = []
    lines.append("=== Explanation utility report ===")
    lines.append(f"config: {args.config}")
    lines.append(f"verdicts (status=ok): {len(summary)}")
    base_err = summary["is_wrong"].mean()
    lines.append(f"base error rate (model wrong): {base_err:.4f}")
    lines.append("")

    # ----- (a) error detection -----
    lines.append("--- (a) error detection: flag model mistakes from verdict ---")
    flag_w = summary["is_wrong"].tolist()
    for name, flagset in [("contradict", {"contradict"}),
                          ("contradict|unclear", {"contradict", "unclear"})]:
        pred = summary["rcl_agreement"].isin(flagset).tolist()
        m = confusion(pred, flag_w)
        lift = m["precision"] / base_err if base_err else float("nan")
        lines.append(f"  flag = {name}:")
        lines.append(f"    confusion  TP={m['tp']} FP={m['fp']} "
                     f"FN={m['fn']} TN={m['tn']}")
        lines.append(f"    precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
                     f"F1={m['f1']:.3f}  specificity={m['specificity']:.3f}")
        lines.append(f"    precision lift over always-trust baseline: {lift:.2f}x")
    lines.append("")
    lines.append("  (baseline 'always trust' flags nothing: recall=0;")
    lines.append("   baseline 'flag everything': precision = base error rate above.)")
    lines.append("")

    # breakdown by true failure type
    lines.append("  contradict-flag precision/recall by true failure type:")
    for ft, grp in summary.groupby("fti_true"):
        pred = (grp["rcl_agreement"] == "contradict").tolist()
        m = confusion(pred, grp["is_wrong"].tolist())
        lines.append(f"    type {ft}: n={len(grp)} wrong={int(grp['is_wrong'].sum())} "
                     f"precision={m['precision']:.3f} recall={m['recall']:.3f}")
    lines.append("")

    # ----- (b) corrective recovery -----
    idx2svc = build_idx_to_service(evidence_dir)
    sample_paths = sorted(expl_dir.glob("sample_*.json"))
    rows, seen = [], set()
    for path in sample_paths:
        with open(path) as f:
            s = json.load(f)
        parsed = s.get("parsed")
        if not parsed:
            continue
        gi = s["gaia_index"]
        if gi in seen or gi not in truth.index:
            continue  # skip duplicate copies / samples without ground truth
        seen.add(gi)
        true_idx = int(truth.loc[gi, "rcl_true"])
        true_svc = idx2svc.get(true_idx)
        pred_svc = s.get("rcl_pred_service")
        text = " ".join([parsed.get("evidence_summary", "") or "",
                         parsed.get("explanation", "") or ""])
        inst, fam = extract_services(text)
        # mention of the true root cause (instance name, or its family stem)
        true_fam = "".join(c for c in (true_svc or "") if not c.isdigit())
        mentions_true = bool(true_svc) and (true_svc in inst or true_fam in fam)
        # redirect = names some service other than the predicted one
        mentions_other = bool(inst - {pred_svc})
        rows.append({
            "gaia_index": gi,
            "group": s.get("group"),
            "rcl_correct": bool(s.get("rcl_correct")),
            "rcl_agreement": parsed.get("rcl_agreement"),
            "pred_service": pred_svc,
            "true_service": true_svc,
            "mentions_true_rc": mentions_true,
            "mentions_pred_rc": pred_svc in inst,
            "mentions_other": mentions_other,
            "n_services_named": len(inst),
        })

    N_INSTANCES = 10  # GAIA service instances -> chance of naming a specific one

    lines.append("--- (b) corrective recovery (needs local sample text) ---")
    if not rows:
        lines.append("  no local sample_*.json with parsed text — skipped.")
    else:
        df = pd.DataFrame(rows)
        lines.append(f"  samples with text: {len(df)}")
        lines.append("")
        lines.append("  corrective recall = among model-WRONG cases in each subset,")
        lines.append("  fraction whose explanation mentions the TRUE root cause.")
        lines.append("  random baseline = mean(k/10), k = #distinct services named")
        lines.append("  (chance of naming the true RC if k services were picked at random).")
        lines.append("")
        lines.append(f"  {'subset':<32} {'n':>4} {'corr.recall':>12} "
                     f"{'random':>8} {'lift':>6}")

        def report_subset(name: str, sub: pd.DataFrame) -> None:
            if not len(sub):
                lines.append(f"  {name:<32} {0:>4}  (empty)")
                return
            cr = sub["mentions_true_rc"].mean()
            rand = (sub["n_services_named"] / N_INSTANCES).mean()
            lift = cr / rand if rand else float("nan")
            lines.append(f"  {name:<32} {len(sub):>4} {cr:>12.3f} "
                         f"{rand:>8.3f} {lift:>6.2f}x")

        wrong = df[~df["rcl_correct"]]
        report_subset("wrong & contradict", wrong[wrong["rcl_agreement"] == "contradict"])
        report_subset("wrong & (contradict|unclear)",
                      wrong[wrong["rcl_agreement"].isin(["contradict", "unclear"])])
        report_subset("all wrong (ceiling, any verdict)", wrong)
        lines.append("")

        # redirect: among wrong & contradict, does the text point elsewhere than pred?
        wc = wrong[wrong["rcl_agreement"] == "contradict"]
        if len(wc):
            lines.append(f"  among 'wrong & contradict' (n={len(wc)}):")
            lines.append(f"    mentions predicted RC: {wc['mentions_pred_rc'].mean():.3f}")
            lines.append(f"    mentions some other service (redirect present): "
                         f"{wc['mentions_other'].mean():.3f}")
            lines.append(f"    mentions TRUE RC specifically: "
                         f"{wc['mentions_true_rc'].mean():.3f}")
        lines.append("")
        lines.append("  note: subsets are conditioned on the LLM verdict, so n shrinks")
        lines.append("  fast (contradict is rare). 'all wrong' is verdict-independent.")

        out_path = out_dir / "utility_per_sample.csv"
        df.to_csv(out_path, index=False)
        lines.append(f"  wrote {out_path}")

    report = "\n".join(lines)
    (out_dir / "utility_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
