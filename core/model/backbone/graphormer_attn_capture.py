"""Step A2 of the interpretability plan.

Captures post-softmax attention weights from a ``GraphormerEncoder``
without altering the original forward pass. The encoder's output stays
bit-for-bit identical; the capture runs as a shadow computation that
re-uses the layer's own ``q_proj`` / ``k_proj`` and the same
``attn_bias`` / ``attn_mask`` arguments.

Why a shadow computation rather than a hook on ``BiasedMHA.forward``:
DGL 2.1.0's ``BiasedMHA`` does not return attention weights and has no
``need_weights`` flag. A pure forward hook would only see inputs and
outputs, not the intermediate softmax. Replacing the layer with our own
implementation risks divergence from DGL's internal math (head
reshape order, scaling, dropout placement). Running an independent
shadow softmax on the same Q, K, attn_bias avoids both problems: it
matches the math the layer performs (verified by row-sum ~ 1.0) while
leaving the actual forward pass untouched.

Usage:

    from core.model.backbone.graphormer_attn_capture import capture_attention

    sink = {}
    with capture_attention(model.MetricGraphormer, sink, name="metric"):
        out = model(...)
    # sink["metric"][layer_idx] -> tensor of shape (B, num_heads, N+1, N+1)
"""
from contextlib import contextmanager
# from types import MethodType
from typing import Dict

import torch


def _shadow_attention_weights(attn_module, nfeat, attn_bias, attn_mask, num_heads):
    """Recompute post-softmax attention weights for a BiasedMHA call.

    Mirrors the math a standard pre-LN multi-head attention performs:
        Q = q_proj(nfeat); K = k_proj(nfeat)
        scores = Q @ K^T / sqrt(d_h) + attn_bias
        weights = softmax(scores)

    nfeat is the *layer-normed* node feature tensor (because GraphormerLayer
    applies attn_layer_norm before calling self.attn when norm_first=True).
    Shape conventions:
        nfeat:     (B, N, F)
        attn_bias: (B, N, N, H)  -> we permute to (B, H, N, N)
        attn_mask: (B, N, N) bool, True = invalid
        weights:   (B, H, N, N)
    """
    B, N, F = nfeat.shape
    H = num_heads
    d_h = F // H

    q = attn_module.q_proj(nfeat).reshape(B, N, H, d_h).permute(0, 2, 1, 3)
    k = attn_module.k_proj(nfeat).reshape(B, N, H, d_h).permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_h ** 0.5)

    if attn_bias is not None:
        scores = scores + attn_bias.permute(0, 3, 1, 2)
    if attn_mask is not None:
        mask_bool = attn_mask if attn_mask.dtype == torch.bool else attn_mask.to(torch.bool)
        scores = scores.masked_fill(mask_bool.unsqueeze(1), float("-inf"))

    return torch.softmax(scores, dim=-1)


@contextmanager
def capture_attention(encoder, sink: Dict, name: str, store_dtype=torch.float16):
    """Patch ``encoder.layers[i].attn.forward`` to record attention.

    Parameters
    ----------
    encoder : GraphormerEncoder
        One of the three modality encoders on MainModel.
    sink : dict
        Mutable dict the wrapper writes into. After ``__exit__``,
        ``sink[name]`` is a list of length ``num_layers``; each entry is
        a tensor of shape ``(B, num_heads, N+1, N+1)`` on CPU.
    name : str
        Modality key (e.g. "metric", "trace", "log").
    store_dtype : torch.dtype
        float16 by default to keep HDF5 storage manageable.

    The wrapper runs only while the context is active. Exiting restores
    the original ``forward`` so the encoder is safe to reuse.
    """
    captured = [None] * len(encoder.layers)
    sink[name] = captured

    originals = []
    num_heads = encoder.num_heads

    for layer_idx, layer in enumerate(encoder.layers):
        attn = layer.attn
        originals.append((attn, attn.forward))

        def make_wrapper(attn_mod, layer_idx=layer_idx, orig=attn.forward):
            def wrapped(nfeat, attn_bias=None, attn_mask=None):
                with torch.no_grad():
                    weights = _shadow_attention_weights(
                        attn_mod, nfeat, attn_bias, attn_mask, num_heads
                    )
                    captured[layer_idx] = weights.to(store_dtype).cpu()
                return orig(nfeat, attn_bias, attn_mask)
            return wrapped

        attn.forward = make_wrapper(attn)

    try:
        yield captured
    finally:
        for attn_mod, orig_forward in originals:
            attn_mod.forward = orig_forward


def stack_captured(sink_per_modality, modalities=("metric", "trace", "log")):
    """Stack a per-modality sink into a single nested dict of stacked tensors.

    Returns
    -------
    dict
        ``{modality: tensor of shape (num_layers, B, num_heads, N+1, N+1)}``
    """
    out = {}
    for m in modalities:
        layers = sink_per_modality.get(m)
        if layers is None or any(t is None for t in layers):
            raise RuntimeError(f"capture for modality '{m}' is incomplete")
        out[m] = torch.stack(layers, dim=0)
    return out


def verify_row_sums(weights, atol=1e-3):
    """Sanity check: post-softmax rows must sum to ~1.0.

    Catches dtype downcasts, wrong dim order, mask issues.
    Run on a single batch immediately after the first capture.
    """
    s = weights.float().sum(dim=-1)
    ok = torch.allclose(s, torch.ones_like(s), atol=atol)
    if not ok:
        max_dev = (s - 1.0).abs().max().item()
        raise AssertionError(
            f"row sums deviate from 1.0 by up to {max_dev:.2e}; "
            "check head dim, mask, or bias permutation"
        )
    return True
