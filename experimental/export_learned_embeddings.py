"""Export post-encoder (learned) multimodal embeddings, out-of-sample.

Companion to extract_attention.py. Where make_embedding_separability.py
projects the *pre-encoder* FastText features (and is contaminated by the
supervised label baked into each node's FastText tag, see
EventProcess.py:77-80), this script takes the representation the model
actually learned: the fused vector

    f = cat(f_m, f_t, f_l)        # 3 * graph_out  (= 96 by default)

that feeds the RCL/FTI heads in MainModel.forward. Every sample is encoded
by the checkpoint of the fold in which it was *held out*, so the resulting
matrix is fully out-of-sample (same 5-disjoint-fold protocol the
explainability pipeline uses). No node is selected by its label, so there
is no RCL leakage.

Output:
    logs/gaia/learned_embeddings/{config}/embeddings.npz
        f          (N, 3*graph_out) float32   -- fused learned embedding
        f_metric   (N, graph_out)   float32
        f_trace    (N, graph_out)   float32
        f_log      (N, graph_out)   float32
        rcl_true   (N,) int         -- root-cause instance index
        fti_true   (N,) int         -- failure-type index
        rcl_pred   (N,) int
        fti_pred   (N,) int
        gaia_index (N,) int
        fold       (N,) int
        rcl_names  (10,) str        -- instance label names (index -> name)
        fti_names  (5,)  str        -- type label names (index -> name)

Run (server, torch+dgl env, GPU optional):
    cd TransTVDiag
    python -m experimental.export_learned_embeddings --config cb_ce_awl
    # add --limit 64 for a fast smoke test on the first 64 samples per fold
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the exact loaders/splits/preset machinery used for attention so the
# embeddings match the checkpoints bit-for-bit.
from experimental.extract_attention import (  # noqa: E402
    CONFIG_PRESETS,
    build_args,
    build_test_loader,
    get_fold_indices,
    load_model,
)
from helper.logger import get_logger  # noqa: E402


def _label_names(dataset, labels_df, fallback_col):
    """Prefer names exposed by the dataset; fall back to sorted gaia.csv."""
    attr = "instance_label_names" if fallback_col == "instance" else "type_label_names"
    names = getattr(dataset, attr, None)
    if names is not None:
        return list(names)
    return sorted(labels_df[fallback_col].astype(str).unique().tolist())


def export(config_name, limit, out_root, device):
    logger = get_logger("logs/gaia", "export_learned_embeddings")

    rows_f, rows_fm, rows_ft, rows_fl = [], [], [], []
    rcl_true, fti_true, rcl_pred, fti_pred = [], [], [], []
    gaia_index, fold_col = [], []
    rcl_names = fti_names = None

    for fold in range(5):
        # reconstruct=True: refit FastText for this fold so the embeddings
        # match what the fold's checkpoint was trained on (same as the
        # default path in extract_attention.py, i.e. no --reuse-embeddings).
        args = build_args(config_name, fold, reconstruct=True)
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")

        labels_df, train_idx, test_idx = get_fold_indices(args, fold)
        if limit is not None:
            test_idx = test_idx[:limit]
        logger.info(f"fold {fold}: train n={len(train_idx)}, held-out n={len(test_idx)}")

        test_dl = build_test_loader(args, train_idx, test_idx, logger)
        model = load_model(args, device)

        if rcl_names is None:
            rcl_names = _label_names(test_dl.dataset, labels_df, "instance")
            fti_names = _label_names(test_dl.dataset, labels_df, "anomaly_type")

        for i, batch in enumerate(test_dl):
            (labels, metric_feat, trace_feat, log_feat,
             in_degree, out_degree, attn_mask, path_data, dist) = batch
            root_true, type_true = labels.flatten().tolist()

            with torch.no_grad():
                (f_m, f_t, f_l), root_logit, type_logit = model(
                    metric_feat.to(device), trace_feat.to(device), log_feat.to(device),
                    in_degree.to(device), out_degree.to(device),
                    dist.to(device), path_data.to(device), attn_mask.to(device),
                )

            f = torch.cat((f_m, f_t, f_l), dim=1).squeeze(0).cpu().numpy()
            rows_f.append(f.astype(np.float32))
            rows_fm.append(f_m.squeeze(0).cpu().numpy().astype(np.float32))
            rows_ft.append(f_t.squeeze(0).cpu().numpy().astype(np.float32))
            rows_fl.append(f_l.squeeze(0).cpu().numpy().astype(np.float32))

            rcl_true.append(int(root_true))
            fti_true.append(int(type_true))
            rcl_pred.append(int(F.softmax(root_logit, -1).argmax().item()))
            fti_pred.append(int(F.softmax(type_logit, -1).argmax().item()))
            gaia_index.append(int(labels_df.iloc[int(test_idx[i])]["index"]))
            fold_col.append(fold)

            if (i + 1) % 200 == 0 or (i + 1) == len(test_dl):
                logger.info(f"  fold {fold}: {i + 1}/{len(test_dl)}")

    out_dir = Path(out_root) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings.npz"
    np.savez_compressed(
        out_path,
        f=np.stack(rows_f),
        f_metric=np.stack(rows_fm),
        f_trace=np.stack(rows_ft),
        f_log=np.stack(rows_fl),
        rcl_true=np.asarray(rcl_true),
        fti_true=np.asarray(fti_true),
        rcl_pred=np.asarray(rcl_pred),
        fti_pred=np.asarray(fti_pred),
        gaia_index=np.asarray(gaia_index),
        fold=np.asarray(fold_col),
        rcl_names=np.asarray([str(x) for x in rcl_names]),
        fti_names=np.asarray([str(x) for x in fti_names]),
    )
    logger.info(f"wrote {out_path}  (N={len(rows_f)}, dim={np.stack(rows_f).shape[1]})")
    print(f"Saved: {out_path}  N={len(rows_f)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="cb_ce_awl", choices=sorted(CONFIG_PRESETS))
    p.add_argument("--limit", type=int, default=None,
                   help="Cap held-out samples per fold for a smoke test")
    p.add_argument("--out-root", default="logs/gaia/learned_embeddings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    export(cli.config, cli.limit, cli.out_root, cli.device)


if __name__ == "__main__":
    main()
