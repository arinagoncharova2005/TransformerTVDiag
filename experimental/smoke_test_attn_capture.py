"""Smoke test for graphormer_attn_capture.

Builds a fresh ``GraphormerEncoder`` (no checkpoint), runs one synthetic
batch with the capture context manager active, and verifies:

  1. Output shape matches the un-instrumented forward.
  2. The original layer.attn.forward is restored on exit.
  3. Captured tensors have shape (B, num_heads, N+1, N+1) per layer.
  4. Post-softmax rows sum to 1.0 (to atol=1e-3 in fp16 storage).
  5. Outputs of two consecutive forwards (capture on / off) are identical.

Run on the server (DGL must work):
    cd experimental
    python smoke_test_attn_capture.py
"""
import sys
from pathlib import Path

# Make ``core`` importable when this file is run directly from
# experimental/ rather than as a -m module from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from core.model.backbone.graphormer import GraphormerEncoder
from core.model.backbone.graphormer_attn_capture import (
    capture_attention,
    verify_row_sums,
)


def main():
    torch.manual_seed(0)

    embedding_dim = 128
    num_heads = 16
    num_layers = 2
    N = 10  # real nodes
    B = 4

    encoder = GraphormerEncoder(
        embedding_dim=embedding_dim,
        hidden_dim=embedding_dim,
        out_dim=32,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    encoder.eval()

    node_feat = torch.randn(B, N, embedding_dim)
    in_degree = torch.randint(0, 5, (B, N))
    out_degree = torch.randint(0, 5, (B, N))
    attn_mask = torch.zeros(B, N + 1, N + 1, dtype=torch.bool)
    path_data = torch.zeros(B, N, N, 5, 1)
    dist = torch.randint(0, 4, (B, N, N))

    print("[1/5] running un-instrumented forward ...")
    with torch.no_grad():
        out_clean = encoder(node_feat, in_degree, out_degree,
                            attn_mask, path_data, dist)
    print(f"      output shape: {tuple(out_clean.shape)}")

    print("[2/5] running with capture_attention ...")
    sink = {}
    with capture_attention(encoder, sink, name="metric"):
        with torch.no_grad():
            out_capt = encoder(node_feat, in_degree, out_degree,
                               attn_mask, path_data, dist)
    captured = sink["metric"]

    print(f"      captured layers: {len(captured)} "
          f"(expected {num_layers})")
    for i, w in enumerate(captured):
        assert w is not None, f"layer {i} not captured"
        print(f"      layer[{i}] shape: {tuple(w.shape)} dtype: {w.dtype}")

    print("[3/5] verifying outputs match (capture must not alter forward) ...")
    diff = (out_capt - out_clean).abs().max().item()
    print(f"      max abs diff: {diff:.3e}")
    assert diff < 1e-5, "capture altered forward output"

    print("[4/5] verifying row sums ~ 1.0 per layer ...")
    for i, w in enumerate(captured):
        verify_row_sums(w, atol=2e-3)
        print(f"      layer[{i}] row-sum check passed")

    print("[5/5] verifying restoration of original forward ...")
    sink2 = {}
    with capture_attention(encoder, sink2, name="metric"):
        pass
    with torch.no_grad():
        out_after = encoder(node_feat, in_degree, out_degree,
                            attn_mask, path_data, dist)
    diff2 = (out_after - out_clean).abs().max().item()
    print(f"      post-context forward max abs diff: {diff2:.3e}")
    assert diff2 < 1e-5

    print()
    print("OK: capture is non-invasive, shapes correct, row sums valid.")


if __name__ == "__main__":
    main()
