"""Step B v2: per-class aggregation of attention.

After Step-B v1 found CLS-row at last layer ≈ uniform across all
classes (a false negative), this version reports three views per sample
per modality:

  1. cls_lift_layer0 / cls_lift_last
       cls_row[rcl_pred] divided by (1 / N_real). 1.0 = uniform,
       higher = CLS concentrates more on the predicted root cause at
       this layer.

  2. cls_argmax_match_layer0 / cls_argmax_match_last
       Whether argmax of the CLS row equals the predicted RCL service.

  3. rcl_outgoing_top1_service_layer0 + rcl_outgoing_top1_attn_layer0
       What service the predicted RCL attends to most (excluding
       itself), and how strongly. This is the dependency-graph view
       — Step-B v1 found e.g. trace encoder layer 0 has dbservice1
       sending 22% + 22% to redisservice1/2, a meaningful pattern that
       CLS aggregation hid.

Outputs (under ``logs/gaia/attention/{config}/aggregates/``):

  per_sample.csv
      One row per (sample, modality). All eight metrics above plus
      sample/class/rcl identifiers. Used by figure scripts and the
      LLM evidence pipeline.

  modality_importance.csv
      Class-level summary: mean of each lift, fraction of samples where
      cls argmax matches RCL, and modal RCL outgoing top-1 service per
      (config, fti_pred, modality). One row per (class, modality).

  {class_name}_{modality}_layer0.npy
  {class_name}_{modality}_last.npy
      Class-level mean cls_row at each layer. Length = N_real.

Notes
-----
* Folds whose artifacts are missing are skipped.
* Class names come from ``h5.attrs["type_names"]`` if present.

Run (from TransTVDiag_experiments_results/):
    python experimental/aggregate_attention.py --config cb_ce_awl
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd

from core.interpret.cls_attention import (
    cls_argmax_matches_rcl,
    cls_lift,
    cls_row_per_modality,
    rcl_outgoing_row,
)


MODALITIES = ("metric", "trace", "log")


def _decode_names(arr) -> List[str]:
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr]


def _load_fold(h5_path: Path, csv_path: Path):
    """Yield (row_dict, captured_dict, type_names, instance_names) per sample."""
    df = pd.read_csv(csv_path)
    with h5py.File(h5_path, "r") as h5:
        type_names = (_decode_names(h5.attrs["type_names"])
                      if "type_names" in h5.attrs else None)
        instance_names = (_decode_names(h5.attrs["instance_names"])
                          if "instance_names" in h5.attrs else None)
        attn = {m: np.asarray(h5[m]["attn_post"], dtype=np.float32)
                for m in MODALITIES}
    n = attn[MODALITIES[0]].shape[1]
    for i in range(n):
        captured = {m: attn[m][:, i] for m in MODALITIES}
        yield df.iloc[i].to_dict(), captured, type_names, instance_names


def _class_name(type_names, class_id: int) -> str:
    if type_names is not None and 0 <= class_id < len(type_names):
        return type_names[class_id]
    return f"class_{class_id}"


def _service_name(instance_names, node_id: int) -> str:
    if instance_names is not None and 0 <= node_id < len(instance_names):
        return instance_names[node_id]
    return f"node_{node_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attention-root", default="logs/gaia/attention")
    ap.add_argument("--out-root", default=None,
                    help="Default: {attention-root}/{config}/aggregates")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    attn_dir = Path(args.attention_root) / args.config
    out_dir = (Path(args.out_root) if args.out_root else attn_dir / "aggregates")
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_sums_l0: Dict[Tuple[str, str], np.ndarray] = {}
    cls_sums_last: Dict[Tuple[str, str], np.ndarray] = {}
    cls_counts: Dict[Tuple[str, str], int] = {}
    per_sample_rows: List[dict] = []
    folds_used = []

    for fold in args.folds:
        h5_path = attn_dir / f"fold_{fold}.h5"
        csv_path = attn_dir / f"fold_{fold}_index.csv"
        if not h5_path.exists() or not csv_path.exists():
            print(f"[skip] fold {fold}: missing artifacts ({h5_path.name})")
            continue

        n_seen = 0
        for row, captured, type_names, instance_names in _load_fold(h5_path, csv_path):
            class_id = int(row["fti_pred"])
            class_name = _class_name(type_names, class_id)
            rcl_pred = int(row["rcl_pred"])

            cls_l0 = cls_row_per_modality(captured, layer=0)
            cls_last = cls_row_per_modality(captured, layer=-1)

            for m in MODALITIES:
                key = (class_name, m)
                if key not in cls_sums_l0:
                    cls_sums_l0[key] = np.zeros_like(cls_l0[m])
                    cls_sums_last[key] = np.zeros_like(cls_last[m])
                    cls_counts[key] = 0
                cls_sums_l0[key] += cls_l0[m]
                cls_sums_last[key] += cls_last[m]
                cls_counts[key] += 1

                out_row = rcl_outgoing_row(captured[m], rcl_pred,
                                           layer=0, mask_self=True)
                top1_node = int(out_row.argmax())

                per_sample_rows.append({
                    "config": args.config,
                    "fold": fold,
                    "sample_idx": int(row["sample_idx"]),
                    "fti_pred": class_name,
                    "rcl_pred": rcl_pred,
                    "rcl_pred_service": _service_name(instance_names, rcl_pred),
                    "modality": m,
                    "cls_lift_layer0": cls_lift(cls_l0[m], rcl_pred),
                    "cls_lift_last": cls_lift(cls_last[m], rcl_pred),
                    "cls_argmax_match_layer0":
                        cls_argmax_matches_rcl(cls_l0[m], rcl_pred),
                    "cls_argmax_match_last":
                        cls_argmax_matches_rcl(cls_last[m], rcl_pred),
                    "rcl_outgoing_top1_node": top1_node,
                    "rcl_outgoing_top1_service":
                        _service_name(instance_names, top1_node),
                    "rcl_outgoing_top1_attn": float(out_row[top1_node]),
                })
            n_seen += 1
        print(f"fold {fold}: aggregated {n_seen} samples")
        folds_used.append(fold)

    if not folds_used:
        raise SystemExit(f"no fold artifacts found under {attn_dir}")

    for (class_name, m), s in cls_sums_l0.items():
        np.save(out_dir / f"{class_name}_{m}_layer0.npy",
                s / cls_counts[(class_name, m)])
        np.save(out_dir / f"{class_name}_{m}_last.npy",
                cls_sums_last[(class_name, m)] / cls_counts[(class_name, m)])
    print(f"wrote {2 * len(cls_sums_l0)} npy files to {out_dir}")

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_path = out_dir / "per_sample.csv"
    per_sample_df.to_csv(per_sample_path, index=False)
    print(f"wrote {per_sample_path}")

    summary = (
        per_sample_df
            .groupby(["config", "fti_pred", "modality"])
            .agg(
                n_samples=("cls_lift_layer0", "size"),
                mean_cls_lift_layer0=("cls_lift_layer0", "mean"),
                mean_cls_lift_last=("cls_lift_last", "mean"),
                frac_cls_argmax_match_layer0=("cls_argmax_match_layer0", "mean"),
                frac_cls_argmax_match_last=("cls_argmax_match_last", "mean"),
                mean_rcl_outgoing_top1_attn=("rcl_outgoing_top1_attn", "mean"),
            )
            .reset_index()
    )

    modal_top1 = (
        per_sample_df
            .groupby(["config", "fti_pred", "modality"])["rcl_outgoing_top1_service"]
            .agg(lambda s: Counter(s).most_common(1)[0][0])
            .reset_index()
            .rename(columns={"rcl_outgoing_top1_service":
                             "modal_rcl_outgoing_top1_service"})
    )
    summary = summary.merge(modal_top1,
                            on=["config", "fti_pred", "modality"], how="left")

    summary_path = out_dir / "modality_importance.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
