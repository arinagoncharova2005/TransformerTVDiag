"""Step A3 of the interpretability plan.

CV-aware attention extraction. Loads one fold's checkpoint, runs
inference on that fold's held-out validation slice, captures the
post-softmax attention weights per modality and per layer, and writes:

    logs/gaia/attention/{config}/fold_{k}.h5
        groups: metric/, trace/, log/
        each contains 'attn_post' shape (num_layers, N, num_heads, M, M)
        where N = len(fold_test_indices), M = num_real_nodes + 1 (CLS).

    logs/gaia/attention/{config}/fold_{k}_index.parquet
        columns: sample_idx, gaia_index, fti_true, fti_pred,
                 rcl_true, rcl_pred, fti_softmax, rcl_softmax

CV protocol matches run_experiments.sh:
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratify by 'instance' (NOT anomaly_type)

Embeddings: by default, refits FastText for the requested fold to match
the embeddings the checkpoint was trained on. For fast iteration during
the MVP, pass --reuse-embeddings to skip the refit and use whatever
data/gaia/tmp/{key}.pkl currently holds.

Run (server, DGL works):
    cd TransTVDiag_experiments_results
    python -m experimental.extract_attention \
        --config cb_ce_awl --fold 0 --limit 64 --reuse-embeddings
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from core.model.MainModel import MainModel
from core.model.backbone.graphormer_attn_capture import (
    capture_attention,
    verify_row_sums,
)
from helper.logger import get_logger
from prepare_data import prepare_for_graphormer
from process.EventProcess import EventProcess


# -----------------------------------------------------------------------------
# Config presets
# -----------------------------------------------------------------------------
# Each entry must produce the same MainModel architecture as training and
# point at the right checkpoint. Filenames match logs/gaia/cv_checkpoints/.

CONFIG_PRESETS = {
    "ce_awl": {
        "root_loss": "ce", "type_loss": "ce",
        "focal_gamma": 2.0, "drw": False, "cb_beta": 0.999,
        "checkpoint": "logs/gaia/cv_checkpoints/ce_awl/"
                      "TransTVDiag-root_ce-type_ce-gamma_2p0-fold_{fold}.pt",
    },
    "focal_awl_gamma_2p0": {
        "root_loss": "focal", "type_loss": "focal",
        "focal_gamma": 2.0, "drw": False, "cb_beta": 0.999,
        "checkpoint": "logs/gaia/cv_checkpoints/focal_awl/"
                      "TransTVDiag-root_focal-type_focal-gamma_2p0-fold_{fold}.pt",
    },
    "cb_ce_awl": {
        "root_loss": "cb_ce", "type_loss": "cb_ce",
        "focal_gamma": 2.0, "drw": False, "cb_beta": 0.999,
        "checkpoint": "logs/gaia/cv_checkpoints/cb_ce_awl/"
                      "TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_{fold}.pt",
    },
}

COMMON_ARGS = dict(
    seed=42,
    dataset="gaia",
    labels_file="gaia.csv",
    N_I=10, N_T=5,
    embedding_dim=128, graph_hidden=128, graph_out=32,
    num_heads=16, num_layers=2,
    seq_hidden=128, linear_hidden=[64],
    feat_drop=0.0, attn_drop=0.0,
    aggregator="lstm",
    modalities="metric,trace,log",
    cv_folds=5, cv_stratify_by="instance",
    TO=True, CM=True,
    dynamic_weight=True,
    epochs=3000, lr=0.001, batch_size=128,
    guide_weight=0.1, aug_percent=0.2, patience=5, temperature=0.3,
    weight_decay=0.0001,
    train=False, evaluate=True,
    reconstruct=True,
    log_step=20, eval_period=10,
    gpu_devices="0",
    experiment_label="",
    model_filename=None,
)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)
    return prepare_for_graphormer(graphs, labels)


def build_args(config_name: str, fold: int, reconstruct: bool) -> SimpleNamespace:
    if config_name not in CONFIG_PRESETS:
        raise SystemExit(
            f"unknown config '{config_name}'. "
            f"choose one of: {sorted(CONFIG_PRESETS)}"
        )
    preset = CONFIG_PRESETS[config_name]
    merged = {**COMMON_ARGS, **{k: v for k, v in preset.items()
                                if k != "checkpoint"}}
    merged["reconstruct"] = reconstruct
    args = SimpleNamespace(**merged)
    args.checkpoint_path = preset["checkpoint"].format(fold=fold)
    args.config_name = config_name
    return args


def get_fold_indices(args, fold: int):
    """Reproduce the StratifiedKFold split used in training."""
    label_path = f"data/{args.dataset}/{args.labels_file}"
    labels_df = pd.read_csv(label_path)
    if args.cv_stratify_by == "combined":
        y = (labels_df["anomaly_type"].astype(str) + "|"
             + labels_df["instance"].astype(str))
    else:
        y = labels_df[args.cv_stratify_by].astype(str)
    splitter = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed,
    )
    splits = list(splitter.split(labels_df, y))
    train_indices, test_indices = splits[fold]
    return labels_df, np.asarray(train_indices), np.asarray(test_indices)


def build_test_loader(args, train_indices, test_indices, logger):
    processor = EventProcess(args, logger)
    _, test_data = processor.process(
        reconstruct=args.reconstruct,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)


def load_model(args, device):
    model = MainModel(args).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"  [warn] missing keys when loading checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys when loading checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model


def stack_layer_list(layer_list):
    """List of length L, each tensor (1, H, M, M) -> tensor (L, H, M, M).

    Squeezes the batch_size=1 dim, since test_dl uses batch_size=1.
    """
    return torch.stack([t.squeeze(0) for t in layer_list], dim=0)


def run_extraction(args, fold: int, limit: int, out_dir: Path, device: str):
    logger = get_logger(f"logs/{args.dataset}", "extract_attention")
    logger.info(f"checkpoint: {args.checkpoint_path}")
    if not Path(args.checkpoint_path).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint_path}")

    labels_df, train_indices, test_indices = get_fold_indices(args, fold)
    if limit is not None:
        test_indices = test_indices[:limit]
        logger.info(f"--limit {limit}: extracting on first {len(test_indices)} fold-test samples")
    logger.info(f"fold {fold}: train n={len(train_indices)}, test n={len(test_indices)}")

    test_dl = build_test_loader(args, train_indices, test_indices, logger)
    model = load_model(args, device)
    instance_names = getattr(test_dl.dataset, "instance_label_names", None)
    type_names = getattr(test_dl.dataset, "type_label_names", None)

    num_layers = args.num_layers
    num_heads = args.num_heads
    M = args.N_I + 1  # virtual node + 10 services

    N = len(test_dl)
    if N == 0:
        raise SystemExit("test loader is empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"fold_{fold}.h5"
    index_path = out_dir / f"fold_{fold}_index.csv"
    logger.info(f"writing attention to {h5_path}")
    logger.info(f"writing index    to {index_path}")

    sample_rows = []
    verified_rowsums = False

    with h5py.File(h5_path, "w") as h5:
        for modality in ("metric", "trace", "log"):
            grp = h5.create_group(modality)
            grp.create_dataset(
                "attn_post",
                shape=(num_layers, N, num_heads, M, M),
                dtype="float16",
                chunks=(num_layers, 1, num_heads, M, M),
                compression="gzip",
                compression_opts=4,
            )
        h5.attrs["config"] = args.config_name
        h5.attrs["fold"] = fold
        h5.attrs["num_samples"] = N
        h5.attrs["num_layers"] = num_layers
        h5.attrs["num_heads"] = num_heads
        if instance_names is not None:
            h5.attrs["instance_names"] = np.array(instance_names, dtype="S")
        if type_names is not None:
            h5.attrs["type_names"] = np.array(type_names, dtype="S")

        for i, batch in enumerate(test_dl):
            (labels, metric_feat, trace_feat, log_feat,
             in_degree, out_degree, attn_mask, path_data, dist) = batch
            root_true, type_true = labels.flatten().tolist()

            sink = {}
            with capture_attention(model.metric_encoder.graph_encoder, sink, "metric"), \
                 capture_attention(model.trace_encoder.graph_encoder, sink, "trace"), \
                 capture_attention(model.log_encoder.graph_encoder, sink, "log"):
                with torch.no_grad():
                    _, root_logit, type_logit = model(
                        metric_feat.to(device), trace_feat.to(device), log_feat.to(device),
                        in_degree.to(device), out_degree.to(device),
                        dist.to(device), path_data.to(device), attn_mask.to(device),
                    )

            for modality in ("metric", "trace", "log"):
                stacked = stack_layer_list(sink[modality])  # (L, H, M, M)
                if not verified_rowsums and modality == "metric":
                    verify_row_sums(stacked.float())
                    verified_rowsums = True
                h5[modality]["attn_post"][:, i, :, :, :] = stacked.numpy().astype(np.float16)

            root_probs = F.softmax(root_logit, dim=-1).flatten().cpu()
            type_probs = F.softmax(type_logit, dim=-1).flatten().cpu()
            root_pred = int(root_probs.argmax().item())
            type_pred = int(type_probs.argmax().item())

            sample_rows.append({
                "sample_idx": i,
                "fold_test_pos": int(test_indices[i]),
                "gaia_index": int(labels_df.iloc[int(test_indices[i])]["index"]),
                "rcl_true": int(root_true),
                "rcl_pred": int(root_pred),
                "rcl_pred_prob": float(root_probs[root_pred].item()),
                "fti_true": int(type_true),
                "fti_pred": int(type_pred),
                "fti_pred_prob": float(type_probs[type_pred].item()),
            })

            if (i + 1) % 100 == 0 or (i + 1) == N:
                logger.info(f"  processed {i + 1}/{N}")

    pd.DataFrame(sample_rows).to_csv(index_path, index=False)
    logger.info("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap fold-test samples for fast iteration; default: all")
    parser.add_argument("--out-root", default="logs/gaia/attention",
                        help="Root dir for HDF5 + parquet output")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Skip FastText refit; use existing data/gaia/tmp/{key}.pkl. "
                             "Use only when those embeddings match the requested fold.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    reconstruct = not cli.reuse_embeddings
    args = build_args(cli.config, cli.fold, reconstruct=reconstruct)
    out_dir = Path(cli.out_root) / cli.config

    run_extraction(args, fold=cli.fold, limit=cli.limit,
                   out_dir=out_dir, device=cli.device)


if __name__ == "__main__":
    main()
