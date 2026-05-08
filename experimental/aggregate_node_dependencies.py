"""Aggregate full node->node attention into a 10x10 dependency graph
per (config, modality, layer), and overlay against the ground-truth
GAIA service-mesh edges.

Step B's per-sample artefacts capture only the *predicted RCL service's*
outgoing top-1 target. That is enough for the 'model agrees with itself'
question but it does not tell us what graph the encoder learned overall
— for that we need every node's outgoing attention, averaged over all
16,205 samples.

For each config the script reads ``logs/gaia/attention/{config}/fold_{k}.h5``
and produces, per (modality, layer), the mean attention matrix ``M[s, t] =
mean over samples of head_mean(attn[layer, s+1, t+1])``. The +1's skip the
CLS row/column. Self-loops are kept (some edges in the GAIA service mesh
are peer edges between two services of the same kind, e.g. log<->log).

It also reads ``data/gaia/events/edges.pkl`` (the static GAIA service
mesh) and reports two summaries per (config, modality, layer):

  - mean attention on ground-truth edges vs mean attention on non-edges
    (off-self). If the encoder learned the mesh, edge-attention should
    be visibly higher than non-edge.

  - top-1 outgoing target per source: did the model pick a real edge
    target?  recall = (# of sources whose argmax target is a true edge)
    / (# of sources with at least one outgoing edge).

Outputs (under ``logs/gaia/attention/{config}/aggregates/``):

  dependency_graph.csv
      long format: config, modality, layer, source, target, mean_attn,
                   is_ground_truth_edge, n_samples_total

  dependency_graph_summary.csv
      one row per (config, modality, layer) with overlap stats

Run (from TransTVDiag_experiments_results/, on the server):
    python experimental/aggregate_node_dependencies.py --config cb_ce_awl
"""
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd


MODALITIES = ("metric", "trace", "log")


def _decode_names(arr) -> List[str]:
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr]


def load_ground_truth_edges(edges_path: Path) -> Set[Tuple[int, int]]:
    """Return {(src_idx, dst_idx)} excluding self-loops."""
    with open(edges_path, "rb") as f:
        edges = pickle.load(f)
    src_list, dst_list = edges[0], edges[1]
    return {(int(s), int(d)) for s, d in zip(src_list, dst_list) if s != d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attention-root", default="logs/gaia/attention")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--edges-pkl", default="data/gaia/events/edges.pkl")
    args = ap.parse_args()

    attn_dir = Path(args.attention_root) / args.config
    out_dir = attn_dir / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_set = load_ground_truth_edges(Path(args.edges_pkl))
    print(f"loaded {len(edges_set)} ground-truth edges (self-loops excluded)")

    instance_names: List[str] = None
    sums = {}
    n_samples = 0

    for fold in args.folds:
        h5_path = attn_dir / f"fold_{fold}.h5"
        if not h5_path.exists():
            print(f"[skip] fold {fold}: missing {h5_path.name}")
            continue
        with h5py.File(h5_path, "r") as h5:
            if instance_names is None and "instance_names" in h5.attrs:
                instance_names = _decode_names(h5.attrs["instance_names"])
            for m in MODALITIES:
                attn = np.asarray(h5[m]["attn_post"], dtype=np.float32)
                num_layers = attn.shape[0]
                head_mean = attn.mean(axis=2)
                node_block = head_mean[:, :, 1:, 1:]      # (L, N, 10, 10)
                fold_sum = node_block.sum(axis=1)         # (L, 10, 10)
                for layer in range(num_layers):
                    key = (m, layer)
                    if key not in sums:
                        sums[key] = np.zeros((10, 10), dtype=np.float64)
                    sums[key] += fold_sum[layer]
            n_samples += attn.shape[1]
        print(f"fold {fold}: accumulated {attn.shape[1]} samples")

    if n_samples == 0:
        raise SystemExit(f"no fold artifacts found under {attn_dir}")
    if instance_names is None:
        instance_names = [f"node_{i}" for i in range(10)]

    long_rows = []
    summary_rows = []
    for (m, layer), s in sorted(sums.items()):
        mean_mat = s / n_samples

        # off-self mask, ground-truth mask
        off_self = ~np.eye(10, dtype=bool)
        gt_mask = np.zeros((10, 10), dtype=bool)
        for (src, dst) in edges_set:
            gt_mask[src, dst] = True

        edge_mean = float(mean_mat[gt_mask].mean()) if gt_mask.any() else 0.0
        non_edge_mask = off_self & ~gt_mask
        non_edge_mean = (float(mean_mat[non_edge_mask].mean())
                         if non_edge_mask.any() else 0.0)

        # argmax target per source (excluding self)
        masked = mean_mat.copy()
        np.fill_diagonal(masked, -np.inf)
        argmax_targets = masked.argmax(axis=1)
        sources_with_edges = sorted({src for (src, _) in edges_set})
        hits = sum(1 for src in sources_with_edges
                   if (src, int(argmax_targets[src])) in edges_set)
        recall_top1 = hits / len(sources_with_edges)

        for src_idx in range(10):
            for dst_idx in range(10):
                long_rows.append({
                    "config": args.config,
                    "modality": m,
                    "layer": layer,
                    "source": instance_names[src_idx],
                    "target": instance_names[dst_idx],
                    "mean_attn": float(mean_mat[src_idx, dst_idx]),
                    "is_ground_truth_edge": (src_idx, dst_idx) in edges_set,
                    "n_samples_total": n_samples,
                })

        summary_rows.append({
            "config": args.config,
            "modality": m,
            "layer": layer,
            "n_samples_total": n_samples,
            "edge_mean_attn": edge_mean,
            "non_edge_mean_attn": non_edge_mean,
            "edge_lift": (edge_mean / non_edge_mean
                          if non_edge_mean > 0 else float("nan")),
            "argmax_top1_recall": recall_top1,
        })

    long_path = out_dir / "dependency_graph.csv"
    summary_path = out_dir / "dependency_graph_summary.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"wrote {long_path} ({len(long_rows)} rows)")
    print(f"wrote {summary_path}")
    print()
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
