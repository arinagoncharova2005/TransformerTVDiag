import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core.model.MainModel import MainModel
from prepare_data import prepare_for_graphormer
from process.EventProcess import EventProcess


class SilentLogger:
    def info(self, msg):
        pass

    def debug(self, msg):
        pass


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)
    return prepare_for_graphormer(graphs, labels)


def plot_confusion_matrix(cm, labels, title, outfile, rotate_x=45):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(Path("data") / args.dataset / args.labels_file)
    instance_names = sorted(labels_df["instance"].unique().tolist())
    failure_type_names = sorted(labels_df["anomaly_type"].unique().tolist())

    args.N_I = len(instance_names)
    args.N_T = len(failure_type_names)

    instance_id_to_name = {idx: name for idx, name in enumerate(instance_names)}
    failure_type_id_to_name = {idx: name for idx, name in enumerate(failure_type_names)}

    test_meta = labels_df[labels_df["data_type"] == "test"].reset_index(drop=True).copy()

    processor = EventProcess(args, SilentLogger())
    _, test_data = processor.process(reconstruct=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    model = MainModel(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records = []
    for sample_idx, batch in enumerate(test_loader):
        labels, metric_feat, trace_feat, log_feat, in_degree, out_degree, attn_mask, path_data, dist = batch
        target_root_idx = labels[:, 0].item()
        target_type_idx = labels[:, 1].item()

        with torch.no_grad():
            _, root_logits, type_logits = model(
                metric_feat.to(device),
                trace_feat.to(device),
                log_feat.to(device),
                in_degree.to(device),
                out_degree.to(device),
                dist.to(device),
                path_data.to(device),
                attn_mask.to(device),
            )

        root_probs = torch.softmax(root_logits, dim=1).cpu()[0]
        type_probs = torch.softmax(type_logits, dim=1).cpu()[0]

        pred_root_idx = int(root_probs.argmax().item())
        pred_type_idx = int(type_probs.argmax().item())

        topk_root = min(5, len(instance_names))
        root_topk_scores, root_topk_indices = torch.topk(root_probs, k=topk_root)
        type_topk_scores, type_topk_indices = torch.topk(type_probs, k=len(failure_type_names))

        meta = test_meta.iloc[sample_idx]
        records.append(
            {
                "sample_position": sample_idx,
                "data_index": int(meta["index"]),
                "datetime": meta["datetime"],
                "service": meta["service"],
                "message": meta["message"],
                "target_root_idx": target_root_idx,
                "target_root_name": instance_id_to_name[target_root_idx],
                "pred_root_idx": pred_root_idx,
                "pred_root_name": instance_id_to_name[pred_root_idx],
                "target_type_idx": target_type_idx,
                "target_type_name": failure_type_id_to_name[target_type_idx],
                "pred_type_idx": pred_type_idx,
                "pred_type_name": failure_type_id_to_name[pred_type_idx],
                "root_correct": pred_root_idx == target_root_idx,
                "type_correct": pred_type_idx == target_type_idx,
                "pred_root_confidence": float(root_probs[pred_root_idx].item()),
                "pred_type_confidence": float(type_probs[pred_type_idx].item()),
                "root_top5_indices": [int(x) for x in root_topk_indices.tolist()],
                "root_top5_names": [instance_id_to_name[int(x)] for x in root_topk_indices.tolist()],
                "root_top5_scores": [float(x) for x in root_topk_scores.tolist()],
                "type_ranked_indices": [int(x) for x in type_topk_indices.tolist()],
                "type_ranked_names": [failure_type_id_to_name[int(x)] for x in type_topk_indices.tolist()],
                "type_ranked_scores": [float(x) for x in type_topk_scores.tolist()],
            }
        )

    predictions_df = pd.DataFrame(records)

    predictions_csv = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)

    root_report = classification_report(
        predictions_df["target_root_name"],
        predictions_df["pred_root_name"],
        labels=instance_names,
        output_dict=True,
        zero_division=0,
    )
    type_report = classification_report(
        predictions_df["target_type_name"],
        predictions_df["pred_type_name"],
        labels=failure_type_names,
        output_dict=True,
        zero_division=0,
    )

    root_report_df = pd.DataFrame(root_report).transpose()
    type_report_df = pd.DataFrame(type_report).transpose()

    root_report_df.to_csv(output_dir / "classification_report_rcl.csv")
    type_report_df.to_csv(output_dir / "classification_report_fti.csv")

    root_cm = confusion_matrix(
        predictions_df["target_root_name"],
        predictions_df["pred_root_name"],
        labels=instance_names,
    )
    type_cm = confusion_matrix(
        predictions_df["target_type_name"],
        predictions_df["pred_type_name"],
        labels=failure_type_names,
    )

    pd.DataFrame(root_cm, index=instance_names, columns=instance_names).to_csv(
        output_dir / "confusion_matrix_rcl.csv"
    )
    pd.DataFrame(type_cm, index=failure_type_names, columns=failure_type_names).to_csv(
        output_dir / "confusion_matrix_fti.csv"
    )

    plot_confusion_matrix(root_cm, instance_names, "RCL Confusion Matrix", output_dir / "confusion_matrix_rcl.png")
    plot_confusion_matrix(
        type_cm, failure_type_names, "FTI Confusion Matrix", output_dir / "confusion_matrix_fti.png", rotate_x=25
    )

    summary = pd.DataFrame(
        [
            {
                "task": "RCL",
                "accuracy": (predictions_df["target_root_name"] == predictions_df["pred_root_name"]).mean(),
                "macro_f1": root_report_df.loc["macro avg", "f1-score"],
                "weighted_f1": root_report_df.loc["weighted avg", "f1-score"],
            },
            {
                "task": "FTI",
                "accuracy": (predictions_df["target_type_name"] == predictions_df["pred_type_name"]).mean(),
                "macro_f1": type_report_df.loc["macro avg", "f1-score"],
                "weighted_f1": type_report_df.loc["weighted avg", "f1-score"],
            },
        ]
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    print(f"Saved predictions to {predictions_csv}")
    print(f"Saved reports and matrices to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TransTVDiag predictions on test set.")
    parser.add_argument("--dataset", type=str, default="gaia", help="Dataset name")
    parser.add_argument("--labels_file", type=str, default="gaia.csv", help="Labels CSV inside dataset folder")
    parser.add_argument("--checkpoint", type=str, default="logs/gaia/TransTVDiag.pt", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="logs/gaia/evaluation_artifacts", help="Where to store outputs")

    # Model hyperparams (must match training)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--graph_hidden", type=int, default=128)
    parser.add_argument("--graph_out", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--linear_hidden", type=list, default=[64])
    parser.add_argument("--aug_percent", type=float, default=0.2)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
