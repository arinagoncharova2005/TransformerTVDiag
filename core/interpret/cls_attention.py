"""Per-sample attention primitives.

Library functions shared by:
  - experimental/aggregate_attention.py (batch per-class aggregation)
  - core/interpret/diagnose.py (online alert path; future)

All functions take captured post-softmax attention from one MainModel
forward pass: ``{modality: array of shape (L, H, M, M)}``, where M = N_real + 1
and CLS sits at index 0.

Two views of the attention are useful:

  cls_row(layer)           — CLS's view of the graph at this layer
                             (drives the [CLS] representation flowing
                             into the FTI/RCL heads).

  rcl_outgoing_row(layer)  — predicted root-cause service's outgoing
                             attention to all real services at this
                             layer (where the dependency-graph story
                             lives, per Step-B v1 inspection).

The CLS view is summarised by two scalars per (sample, modality, layer):

  cls_lift = cls_row[rcl_pred] / (1 / N_real)      — concentration on
      the predicted RCL relative to a uniform attention baseline.
      lift = 1 means uniform; lift > 1 means concentrated.

  cls_argmax_matches_rcl                           — does the most-attended
      service equal the predicted RCL? boolean.
"""
from typing import Dict

import numpy as np


def cls_row_per_modality(
    captured: Dict[str, np.ndarray],
    layer: int = -1,
) -> Dict[str, np.ndarray]:
    """CLS->real-services attention per modality at the chosen layer.

    Returns ``{modality: ndarray of shape (N_real,)}``: mean over heads of
    ``attn[layer, 0, 1:]``.
    """
    out: Dict[str, np.ndarray] = {}
    for modality, raw in captured.items():
        a = np.asarray(raw, dtype=np.float32)
        if a.ndim != 4:
            raise ValueError(
                f"expected (L, H, M, M) for modality '{modality}', got {a.shape}"
            )
        cls_row = a[layer].mean(axis=0)[0, 1:]
        out[modality] = cls_row
    return out


def cls_lift(cls_row: np.ndarray, rcl_pred_node: int) -> float:
    """Lift of CLS attention on the predicted RCL over uniform baseline.

    Returns ``cls_row[rcl_pred] * N_real``. 1.0 = uniform.
    """
    return float(cls_row[rcl_pred_node] * len(cls_row))


def cls_argmax_matches_rcl(cls_row: np.ndarray, rcl_pred_node: int) -> bool:
    """Whether the CLS-most-attended service equals the predicted RCL."""
    return int(cls_row.argmax()) == int(rcl_pred_node)


def rcl_outgoing_row(
    captured_modality: np.ndarray,
    rcl_pred_node: int,
    layer: int = 0,
    mask_self: bool = True,
) -> np.ndarray:
    """The predicted RCL service's outgoing attention to other services.

    Parameters
    ----------
    captured_modality : array
        Shape ``(L, H, M, M)``.
    rcl_pred_node : int
        Index in ``[0, N_real)`` of the predicted root-cause service.
    layer : int
        Layer index. Default 0 — Step-B v1 inspection found node->node
        structure is most pronounced at layer 0; layer 1 is smoothed.
    mask_self : bool
        If True, set the self-attention entry (rcl_pred -> rcl_pred) to 0
        before returning, so an argmax over the row picks the most-attended
        *other* service.

    Returns
    -------
    ndarray of shape ``(N_real,)``
        Head-mean of ``attn[layer, rcl_pred + 1, 1:]``. The +1 skips the
        CLS row; the [1:] skips the CLS column.
    """
    a = np.asarray(captured_modality, dtype=np.float32)
    if a.ndim != 4:
        raise ValueError(f"expected (L, H, M, M), got {a.shape}")
    head_mean = a[layer].mean(axis=0)            # (M, M)
    out = head_mean[rcl_pred_node + 1, 1:].copy()
    if mask_self:
        out[rcl_pred_node] = 0.0
    return out
