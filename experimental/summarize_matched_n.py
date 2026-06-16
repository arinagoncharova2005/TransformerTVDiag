"""Summarise the matched-N experiment straight from the model's own test
predictions -- no probe, no embedding export.

The CV run writes one `*_test_predictions.csv` per fold (target_root_name,
pred_root_name, ...). Pooled over folds these cover every sample out-of-fold,
so per-instance RCL recall on the full data is read directly from the model.

Reports per-instance recall, precision and F1 (recall alone can mislead: a
balanced model may over-predict a class, inflating its recall while precision
falls). All classes were trained on ~equal small N (majority capped to the
rare size). Compare the majority instances (mobservice1/2) with the rare ones,
using F1 as the headline.
  - majority F1 still high, rare F1 still low -> the gap is the signal, not N.
  - majority F1 drops to the rare level       -> the gap is sample size.
Note: with balanced training the model loses its majority prior, so the
*absolute* majority recall drops a bit on its own; read the majority-vs-rare
GAP at equal N, not the absolute number.

Run (anywhere with pandas):
    python experimental/summarize_matched_n.py \
        --artifacts_dir logs/gaia/cv_evaluation_artifacts/cb_ce_awl_cap50
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

MAJORITY = {"mobservice1", "mobservice2"}


def plot_confusion(y_true, y_pred, order, support, out_path):
    """Row-normalised RCL confusion (recall view): for each true instance, the
    fraction of its samples routed to each predicted instance. Rows/cols are
    ordered by support (majority first). Clean Blues colormap."""
    cm = confusion_matrix(y_true, y_pred, labels=order).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    labels = [f"{name}\n(n={n})" for name, n in zip(order, support)]

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted instance")
    ax.set_ylabel("true instance")
    ax.set_title("RCL confusion under matched-N training (row-normalised)")
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.5 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved confusion matrix: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", required=True,
                   help="dir with the per-fold *_test_predictions.csv files")
    cli = p.parse_args()

    files = sorted(glob.glob(os.path.join(cli.artifacts_dir, "*_test_predictions.csv")))
    if not files:
        raise SystemExit(f"no *_test_predictions.csv in {cli.artifacts_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"pooled {len(files)} folds, {len(df)} samples\n")

    y_true = df["target_root_name"]
    y_pred = df["pred_root_name"]
    names = sorted(y_true.unique())
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=names, average=None, zero_division=0
    )
    rows = sorted(zip(names, support, rec, prec, f1), key=lambda r: -r[1])

    print(f"{'instance':<16}{'n':>7}{'recall':>9}{'prec':>9}{'f1':>9}   group")
    for inst, n, r, p, f in rows:
        group = "majority" if inst in MAJORITY else "rare"
        print(f"  {inst:<14}{n:>7}{r:>9.3f}{p:>9.3f}{f:>9.3f}   {group}")

    def mean(metric_idx, majority):
        vals = [row[metric_idx] for row in rows
                if (row[0] in MAJORITY) == majority]
        return sum(vals) / len(vals)

    print()
    print(f"majority  recall {mean(2, True):.3f}  prec {mean(3, True):.3f}  f1 {mean(4, True):.3f}")
    print(f"rare      recall {mean(2, False):.3f}  prec {mean(3, False):.3f}  f1 {mean(4, False):.3f}")
    print("\nCompare the majority-vs-rare F1 gap at equal N:")
    print("  majority F1 >> rare F1 -> the gap is the signal (not N);")
    print("  majority F1 ~= rare F1 -> the gap is sample size.")
    print("(recall alone can mislead: balanced training drops the majority prior,"
          " so read precision/F1 too.)")

    order = [r[0] for r in rows]            # by support, majority first
    support_ordered = [r[1] for r in rows]
    plot_confusion(y_true, y_pred, order, support_ordered,
                   os.path.join(cli.artifacts_dir, "rcl_confusion_matched_n.png"))


if __name__ == "__main__":
    main()
