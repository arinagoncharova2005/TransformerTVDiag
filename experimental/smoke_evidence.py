"""Smoke test for core/interpret/evidence.py.

Picks a sample with the strongest trace cls_lift on cb_ce_awl and one
that is covered by the log_per_sample_lookup (so we get real raw log
lines in the evidence). Builds the bundle and prints it as JSON.

Run (from TransTVDiag_attention_analysis/):
    python experimental/smoke_evidence.py
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd

from core.interpret.evidence import (
    load_event_files,
    load_log_lookup,
    build_evidence_bundle,
)


CONFIG = "cb_ce_awl"
ATTN_ROOT = Path("logs/gaia/attention")
EVENTS_DIR = Path("data/gaia/events")


def main():
    print(f"loading events from {EVENTS_DIR}...")
    events = load_event_files(EVENTS_DIR)
    lookup = load_log_lookup(EVENTS_DIR / "log_per_sample_lookup.json")
    print(f"  log: {len(events['log'])} samples; trace: {len(events['trace'])}; "
          f"metric: {len(events['metric'])}")
    print(f"  log_lookup: {len(lookup['samples'])} samples covered, "
          f"{len(lookup['templates'])} templates")

    # Find a sample with high trace lift, then resolve its gaia_index from
    # the fold index csv (only the lookup is keyed by gaia_index).
    aggr = pd.read_csv(ATTN_ROOT / CONFIG / "aggregates" / "per_sample.csv")
    aggr = aggr[aggr["modality"] == "trace"]
    if aggr.empty:
        raise SystemExit("no trace rows found in per_sample.csv")
    aggr = aggr.sort_values("cls_lift_layer0", ascending=False)

    target = None
    pos_in_fold = None
    fold_df = None
    for _, candidate in aggr.iterrows():
        sid_c = int(candidate["sample_idx"])
        fold_c = int(candidate["fold"])
        csv_c = ATTN_ROOT / CONFIG / f"fold_{fold_c}_index.csv"
        df_c = pd.read_csv(csv_c).reset_index(drop=True)
        idx_match = df_c.index[df_c["sample_idx"] == sid_c]
        if len(idx_match) == 0:
            continue
        pos_c = int(idx_match[0])
        gid_c = int(df_c.iloc[pos_c]["gaia_index"])
        if str(gid_c) in lookup["samples"]:
            target = candidate
            pos_in_fold = pos_c
            fold_df = df_c
            break
    if target is None:
        raise SystemExit("no candidate with both attention and lookup coverage")

    sid = int(target["sample_idx"])
    fold = int(target["fold"])
    gaia_index = int(fold_df.iloc[pos_in_fold]["gaia_index"])
    print()
    print(f"target: sample_idx={sid}, gaia_index={gaia_index}, fold={fold}")
    print(f"  fti_pred={target['fti_pred']}, rcl_pred_service={target['rcl_pred_service']}")
    print(f"  trace cls_lift_layer0={target['cls_lift_layer0']:.3f}, "
          f"argmax_match={target['cls_argmax_match_layer0']}")
    print(f"  position in fold: {pos_in_fold}")

    h5_path = ATTN_ROOT / CONFIG / f"fold_{fold}.h5"

    with h5py.File(h5_path, "r") as h5:
        instance_names = [
            n.decode() if isinstance(n, (bytes, bytearray)) else str(n)
            for n in h5.attrs["instance_names"]
        ]
        cls_attns = {}
        for m in ("metric", "trace", "log"):
            attn = np.asarray(h5[m]["attn_post"][0, pos_in_fold],
                              dtype=np.float32)
            cls_attns[m] = attn.mean(axis=0)[0, 1:]

    bundle = build_evidence_bundle(
        sample_idx=sid,
        gaia_index=gaia_index,
        cls_attns=cls_attns,
        rcl_pred=int(target["rcl_pred"]),
        fti_pred=str(target["fti_pred"]),
        instance_names=instance_names,
        events=events,
        log_lookup=lookup,
        top_k=5,
    )

    out_path = Path("logs/gaia/evidence_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print()
    print(f"wrote {out_path}")
    print()
    print("--- bundle (truncated to first 4000 chars) ---")
    text = json.dumps(bundle, indent=2, ensure_ascii=False)
    print(text[:4000])
    if len(text) > 4000:
        print(f"... [{len(text) - 4000} more chars in {out_path}]")


if __name__ == "__main__":
    main()
