"""Build per-sample raw-log lookup for the evidence pipeline.

One-off, server-side. For each sample (incident window) we already
have the raw ``log`` DataFrame in the pickled shards under
``/GAIA_data/data/gaia/pkl/split/`` — that is the same source the
original log_event_extractor consumed when generating
``data/gaia/events/log.json``.

This script reads each shard, classifies every raw log line in every
sample with the trained Drain miner (``extractor/drain/drain.pkl``),
and writes — per sample, per (service, template_id) — up to
``--max-examples`` raw text lines from that sample's own incident
window. Optionally a timestamp is kept alongside each line.

The point: for sample S where the model attended to (logservice2,
template_id=23), the evidence package shows the *actual* raw log
lines from logservice2 inside S's incident window that matched
template 23. Not arbitrary corpus examples.

Output JSON layout::

    {
      "templates": {"<template_id>": "<template text>"},
      "samples": {
        "<sample_idx>": {
          "<service>__<template_id>": [
            {"timestamp": "...", "message": "raw line"},
            ...
          ],
          ...
        },
        ...
      }
    }

Run (from TransTVDiag_experiments_results/, on the server):
    python experimental/build_per_sample_log_lookup.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from extractor.utils import io_util


DEFAULT_DRAIN_PKL = "extractor/drain/drain.pkl"
DEFAULT_STATS_CSV = "extractor/drain/statistics.csv"
DEFAULT_PKL_SPLIT_DIR = "/GAIA_data/data/gaia/pkl/split"
DEFAULT_OUT_PATH = "data/gaia/events/log_per_sample_lookup.json"


def _line_to_record(message, timestamp):
    rec = {"message": str(message).rstrip()}
    if timestamp is not None and not pd.isna(timestamp):
        rec["timestamp"] = str(timestamp)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drain-pkl", default=DEFAULT_DRAIN_PKL)
    ap.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    ap.add_argument("--pkl-split-dir", default=DEFAULT_PKL_SPLIT_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    ap.add_argument("--max-examples", type=int, default=3,
                    help="Raw lines kept per (sample, service, template_id)")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="Process only the first N shards (smoke test)")
    ap.add_argument("--limit-samples", type=int, default=None,
                    help="Stop after this many samples have been processed "
                         "(across shards). For quick smoke tests.")
    args = ap.parse_args()

    print(f"loading miner from {args.drain_pkl}")
    miner = io_util.load(args.drain_pkl)
    stats = pd.read_csv(args.stats_csv)
    template_text = {int(tid): tpl for tid, tpl in
                     zip(stats["id"], stats["template"])}
    print(f"miner has {len(miner.drain.clusters)} clusters; "
          f"statistics.csv has {len(stats)} templates")

    samples_out: dict = {}

    shards = sorted(os.listdir(args.pkl_split_dir))
    if args.limit_shards is not None:
        shards = shards[:args.limit_shards]
    print(f"processing {len(shards)} shard(s)")

    samples_seen = 0
    for shard_name in shards:
        shard_path = os.path.join(args.pkl_split_dir, shard_name)
        print(f"  loading {shard_name} ...", flush=True)
        data = io_util.load(shard_path)

        stop = False
        for sample_idx, sample in data.items():
            log_df = sample.get("log")
            if log_df is None or len(log_df) == 0:
                continue
            services = log_df["service"].values
            messages = log_df["message"].values
            timestamps = (log_df["timestamp"].values
                          if "timestamp" in log_df.columns
                          else [None] * len(log_df))

            sample_bucket: dict = {}
            for svc, msg, ts in zip(services, messages, timestamps):
                cluster = miner.match(msg)
                if cluster is None:
                    continue
                tid = int(cluster.cluster_id)
                key = f"{svc}__{tid}"
                bucket = sample_bucket.setdefault(key, [])
                if len(bucket) < args.max_examples:
                    bucket.append(_line_to_record(msg, ts))
            if sample_bucket:
                samples_out[str(int(sample_idx))] = sample_bucket

            samples_seen += 1
            if (args.limit_samples is not None
                    and samples_seen >= args.limit_samples):
                stop = True
                break

        del data
        if stop:
            print(f"reached --limit-samples={args.limit_samples}, stopping")
            break

    payload = {
        "templates": {str(tid): txt for tid, txt in template_text.items()},
        "samples": samples_out,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_samples = len(samples_out)
    n_pairs = sum(len(v) for v in samples_out.values())
    n_lines = sum(len(lines) for v in samples_out.values()
                  for lines in v.values())
    print()
    print(f"wrote {out_path}")
    print(f"  samples covered:           {n_samples}")
    print(f"  (sample, svc, tid) buckets: {n_pairs}")
    print(f"  raw log lines kept:         {n_lines}")


if __name__ == "__main__":
    main()
