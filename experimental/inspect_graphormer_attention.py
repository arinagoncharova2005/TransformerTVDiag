"""Step A1 of the interpretability plan.

Run this once on the server (where DGL works) to confirm how
``dgl.nn.GraphormerLayer`` exposes its attention internals. The output
of this script determines whether the attention-capture context manager
in step A2 uses a forward hook on an inner attention sub-module, or a
monkey-patch around ``layer.forward`` itself.

What we want to know:

1.  Does ``GraphormerLayer.forward`` have a ``need_weights`` (or similar)
    keyword argument that returns post-softmax attention?
2.  Is there a sub-module on the layer that *is* a standard attention
    block (e.g. ``nn.MultiheadAttention``)? If yes, register a forward
    hook on it.
3.  If neither, what does the source of ``GraphormerLayer.forward`` look
    like? We'll mirror the math (Q @ K^T / sqrt(d) + attn_bias, softmax)
    inside the capture wrapper.

Run:
    cd TransTVDiag_experiments_results
    python -m experimental.inspect_graphormer_attention
"""
import inspect
import sys

import dgl
import torch
from dgl.nn import GraphormerLayer


def main():
    print(f"Python:    {sys.version.split()[0]}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"DGL:       {dgl.__version__}")
    print()

    layer = GraphormerLayer(
        feat_size=128,
        hidden_size=128,
        num_heads=16,
        dropout=0.1,
        norm_first=True,
    )

    print("=" * 70)
    print("GraphormerLayer.forward signature")
    print("=" * 70)
    print(inspect.signature(layer.forward))
    print()

    print("=" * 70)
    print("GraphormerLayer.forward source")
    print("=" * 70)
    try:
        print(inspect.getsource(layer.forward))
    except (OSError, TypeError) as e:
        print(f"  (source not available: {e})")
    print()

    print("=" * 70)
    print("Top-level submodules on GraphormerLayer")
    print("=" * 70)
    for name, mod in layer.named_children():
        print(f"  {name:>20s}  ::  {type(mod).__name__}")
    print()

    print("=" * 70)
    print("All parameter names on the layer")
    print("=" * 70)
    for name, _ in layer.named_parameters():
        print(f"  {name}")
    print()

    print("=" * 70)
    print("Attention sub-module deep-dive")
    print("=" * 70)
    candidates = []
    for name, mod in layer.named_modules():
        cls_name = type(mod).__name__
        if any(k in cls_name.lower() for k in ("attention", "attn", "multihead")):
            candidates.append((name, mod, cls_name))
    if candidates:
        for name, mod, cls_name in candidates:
            print(f"  found:  {name or '<root>'}  ({cls_name})")
            try:
                print(f"          forward signature: {inspect.signature(mod.forward)}")
            except (ValueError, TypeError):
                print("          forward signature: (unavailable)")
    else:
        print("  no submodule matches *attention*/*attn*/*multihead* in class name")
    print()

    print("=" * 70)
    print("Smoke test forward (B=2, N=11, F=128) — confirms shapes")
    print("=" * 70)
    B, N, F = 2, 11, 128
    H = 16
    x = torch.randn(B, N, F)
    attn_bias = torch.zeros(B, N, N, H)
    layer.eval()
    with torch.no_grad():
        try:
            out = layer(x, attn_bias=attn_bias)
        except TypeError as e:
            print(f"  forward(x, attn_bias=...) raised: {e}")
            print("  retrying with positional only ...")
            out = layer(x, attn_bias)
    print(f"  output shape: {tuple(out.shape)}  (expected ({B}, {N}, {F}))")
    print()

    print("=" * 70)
    print("Decision matrix (read manually from the output above):")
    print("=" * 70)
    print("  - If 'need_weights' or 'return_attention' appears in the forward")
    print("    signature: A2 uses that flag directly. EASIEST.")
    print()
    print("  - Else if there is an attention sub-module (e.g. 'MultiheadAttention'")
    print("    or DGL-internal MHA): register a forward hook on it. SECOND BEST.")
    print()
    print("  - Else (attention is fused inline in GraphormerLayer.forward source):")
    print("    monkey-patch GraphormerLayer.forward to mirror Q@K^T + attn_bias,")
    print("    softmax, and stash the post-softmax tensor on a side dict.")


if __name__ == "__main__":
    main()
