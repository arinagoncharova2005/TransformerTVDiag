"""Batch runner for Step C — generate per-modality evidence bundles for
every sample with extracted attention.

For each fold of a given config (default cb_ce_awl) reads:
  - logs/gaia/attention/<config>/fold_{k}.h5
  - logs/gaia/attention/<config>/fold_{k}_index.csv

For each sample in the fold, pulls layer-0 CLS attention (head-mean,
[CLS] row, real-node columns) for metric/trace/log encoders, then calls
build_evidence_bundle_per_modality to assemble a structured bundle that
always contains all three modalities (empty event lists kept as-is).

Output: logs/gaia/evidence/<config>/evidence_fold_{k}.jsonl — one JSON
object per line.

Run (from TransTVDiag_attention_analysis/):
    python experimental/run_evidence_batch.py
    python experimental/run_evidence_batch.py --config ce_awl
    python experimental/run_evidence_batch.py --folds 0,1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd

from core.interpret.evidence import (
    load_event_files,
    load_log_lookup,
    build_evidence_bundle_per_modality,
)


MODALITIES = ("metric", "trace", "log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="cb_ce_awl",
                   help="attention config subdir under logs/gaia/attention/")
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="comma-separated fold indices")
    p.add_argument("--top_k_per_modality", type=int, default=3,
                   help="how many top services per modality to record")
    p.add_argument("--layer", type=int, default=0,
                   help="which encoder layer's attention to use (default: 0)")
    p.add_argument("--attn_root", default="logs/gaia/attention")
    p.add_argument("--events_dir", default="data/gaia/events")
    p.add_argument("--out_root", default="logs/gaia/evidence")
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def process_fold(
    fold: int,
    config_dir: Path,
    out_dir: Path,
    events: dict,
    lookup: Optional[dict],
    top_k: int,
    layer: int,
    log_every: int,
) -> dict:
    """Process one fold; return per-fold counters."""
    h5_path = config_dir / f"fold_{fold}.h5"
    csv_path = config_dir / f"fold_{fold}_index.csv"
    out_path = out_dir / f"evidence_fold_{fold}.jsonl"

    if not h5_path.exists():
        print(f"  SKIP fold {fold}: {h5_path} not found")
        return {"fold": fold, "written": 0, "skipped": True}
    if not csv_path.exists():
        print(f"  SKIP fold {fold}: {csv_path} not found")
        return {"fold": fold, "written": 0, "skipped": True}

    fold_df = pd.read_csv(csv_path).reset_index(drop=True)
    n_samples = len(fold_df)
    print(f"\nfold {fold}: {n_samples} samples — reading {h5_path.name}")

    with h5py.File(h5_path, "r") as h5, open(out_path, "w") as fout:
        instance_names = [
            (n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
            for n in h5.attrs["instance_names"]
        ]
        attn_datasets = {m: h5[m]["attn_post"] for m in MODALITIES}

        n_lookup_hits = 0
        n_with_any_log_raw = 0

        for pos in range(n_samples):
            row = fold_df.iloc[pos].to_dict()
            cls_attns = {}
            for m in MODALITIES:
                attn = np.asarray(attn_datasets[m][layer, pos], dtype=np.float32)
                cls_attns[m] = attn.mean(axis=0)[0, 1:]

            sample_idx = int(row["sample_idx"])
            gaia_index = int(row["gaia_index"])
            bundle = build_evidence_bundle_per_modality(
                sample_idx=sample_idx,
                gaia_index=gaia_index,
                cls_attns=cls_attns,
                rcl_pred=int(row["rcl_pred"]),
                fti_pred=str(row.get("fti_pred", "")),
                instance_names=instance_names,
                events=events,
                log_lookup=lookup,
                top_k_per_modality=top_k,
            )
            bundle["fold"] = fold

            if lookup is not None and str(gaia_index) in lookup["samples"]:
                n_lookup_hits += 1
            for log_entry in bundle["evidence_by_modality"].get("log", []):
                tpls = log_entry.get("details", {}).get("templates", [])
                if any("raw_examples" in t for t in tpls):
                    n_with_any_log_raw += 1
                    break

            fout.write(json.dumps(bundle, ensure_ascii=False) + "\n")

            if (pos + 1) % log_every == 0:
                print(f"  ...{pos + 1}/{n_samples}")

    print(f"  wrote {out_path}")
    print(f"  lookup hits: {n_lookup_hits}/{n_samples} "
          f"({100 * n_lookup_hits / max(1, n_samples):.1f}%)")
    print(f"  samples with at least one raw log example: "
          f"{n_with_any_log_raw}/{n_samples}")
    return {
        "fold": fold,
        "written": n_samples,
        "lookup_hits": n_lookup_hits,
        "with_raw_log": n_with_any_log_raw,
        "out_path": str(out_path),
    }


def main() -> None:
    args = parse_args()
    folds = [int(x) for x in args.folds.split(",")]

    print(f"loading events from {args.events_dir}...")
    events = load_event_files(Path(args.events_dir))
    lookup = load_log_lookup(
        Path(args.events_dir) / "log_per_sample_lookup.json")
    if lookup is None:
        print("WARNING: log_per_sample_lookup.json missing — "
              "raw_examples will be absent")
    else:
        print(f"  log_lookup: {len(lookup['samples'])} samples covered, "
              f"{len(lookup['templates'])} templates")
    print(f"  events: log={len(events['log'])} samples, "
          f"trace={len(events['trace'])}, metric={len(events['metric'])}")

    config_dir = Path(args.attn_root) / args.config
    out_dir = Path(args.out_root) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nconfig: {args.config}")
    print(f"  in:  {config_dir}")
    print(f"  out: {out_dir}")
    print(f"  top_k_per_modality={args.top_k_per_modality}, layer={args.layer}")

    summaries = []
    for fold in folds:
        summaries.append(process_fold(
            fold=fold,
            config_dir=config_dir,
            out_dir=out_dir,
            events=events,
            lookup=lookup,
            top_k=args.top_k_per_modality,
            layer=args.layer,
            log_every=args.log_every,
        ))

    summary_path = out_dir / "batch_summary.json"
    summary_payload = {
        "config": args.config,
        "top_k_per_modality": args.top_k_per_modality,
        "layer": args.layer,
        "folds": summaries,
        "total_written": sum(s.get("written", 0) for s in summaries),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"\nwrote summary: {summary_path}")
    print(f"total bundles: {summary_payload['total_written']} "
          f"across {len(folds)} folds")


if __name__ == "__main__":
    main()
