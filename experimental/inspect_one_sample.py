"""Diagnose attention structure on a single sample.

Step B revealed that class-level CLS-row attention looks essentially
uniform (mean_attn_to_rcl_pred 0.089-0.099 against a 0.1 uniform
baseline). Before committing to interpretability findings, this script
inspects one sample's full attention matrices to answer:

  1. Is structure present at layer 0 but smoothed at layer 1?
  2. Is structure present in node->node attention but not in cls_row?
  3. Is the metric encoder degenerate (uniform softmax everywhere)?
  4. Or is everything uniform everywhere?

For each modality and each layer, prints:
  - the full (11, 11) attention matrix, mean over heads
  - the cls_row = W[0, 1:] (length 10)
  - per-row entropy (uniform-over-11 baseline = log(11) ~ 2.398 nats)

Run (from TransTVDiag_experiments_results/):
    python experimental/inspect_one_sample.py --config cb_ce_awl --fold 0 --sample-idx 0
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd


MODALITIES = ("metric", "trace", "log")


def _row_entropy(p):
    """Per-row Shannon entropy in nats. Safe against zeros."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _decode_names(arr):
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--attention-root", default="logs/gaia/attention")
    args = ap.parse_args()

    h5_path = Path(args.attention_root) / args.config / f"fold_{args.fold}.h5"
    csv_path = Path(args.attention_root) / args.config / f"fold_{args.fold}_index.csv"
    if not h5_path.exists() or not csv_path.exists():
        raise SystemExit(f"missing artifacts at {h5_path}")

    df = pd.read_csv(csv_path)
    row = df.iloc[args.sample_idx].to_dict()
    print(f"sample_idx     : {args.sample_idx}")
    print(f"gaia_index     : {row['gaia_index']}")
    print(f"fti_true / pred: {row['fti_true']} / {row['fti_pred']} (p={row['fti_pred_prob']:.3f})")
    print(f"rcl_true / pred: {row['rcl_true']} / {row['rcl_pred']} (p={row['rcl_pred_prob']:.3f})")

    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    with h5py.File(h5_path, "r") as h5:
        type_names = (_decode_names(h5.attrs["type_names"])
                      if "type_names" in h5.attrs else None)
        instance_names = (_decode_names(h5.attrs["instance_names"])
                          if "instance_names" in h5.attrs else None)
        if type_names is not None:
            print(f"FTI label name : {type_names[int(row['fti_pred'])]}")
        if instance_names is not None:
            print(f"RCL pred name  : {instance_names[int(row['rcl_pred'])]}")

        for m in MODALITIES:
            attn = np.asarray(h5[m]["attn_post"][:, args.sample_idx, :, :, :],
                              dtype=np.float32)
            num_layers, num_heads, M, _ = attn.shape
            print()
            print("=" * 78)
            print(f"modality: {m}    shape: layers={num_layers} heads={num_heads} M={M}")
            print("=" * 78)
            uniform_h = float(np.log(M))
            print(f"uniform-over-{M} entropy baseline: {uniform_h:.3f} nats")

            for L in range(num_layers):
                A = attn[L].mean(axis=0)              # (M, M)
                cls_row = A[0, 1:]                    # (M-1,)
                entropies = _row_entropy(A)           # (M,)

                print()
                print(f"-- layer {L} (mean over heads) --")
                print(f"matrix:")
                print(A)
                print(f"row sums: {A.sum(axis=-1)}")
                print(f"cls_row (W[0,1:]): {cls_row}")
                print(f"  cls_row max/min : {cls_row.max():.3f} / {cls_row.min():.3f}")
                if instance_names is not None:
                    argmax_node = int(cls_row.argmax())
                    print(f"  cls_row argmax  : {argmax_node} ({instance_names[argmax_node]})")
                print(f"per-row entropy   : {entropies}")
                print(f"  rows below 1.0 nat (concentrated): "
                      f"{int((entropies < 1.0).sum())} / {M}")
                print(f"  mean row entropy / uniform baseline: "
                      f"{entropies.mean():.3f} / {uniform_h:.3f}")


if __name__ == "__main__":
    main()
