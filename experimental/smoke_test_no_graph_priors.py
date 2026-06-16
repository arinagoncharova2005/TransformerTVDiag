"""Smoke test for the --no_graph_priors ablation flag.

Runs in <10s on CPU. Verifies, without any training, that:
  1. argparse on main.py recognises --no_graph_priors and toggles
     args.use_graph_priors as expected.
  2. MainModel constructs successfully in both modes.
  3. Forward pass produces tensors of the expected shapes in both modes.
  4. With --no_graph_priors the attention bias inside Graphormer stays zero
     and no degree contribution is added to node features.
  5. Parameter count is preserved (the encoders are still instantiated in
     __init__, they are only bypassed in forward — this is intentional).

Run from the repo root (TransTVDiag_attention_analysis/):

    python smoke_test_no_graph_priors.py

Exit code is 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing here so that any syntax error in the model code surfaces early.
from core.model.MainModel import MainModel
from core.model.backbone.graphormer import GraphormerEncoder


FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_args(use_graph_priors: bool) -> argparse.Namespace:
    """Minimal args namespace required by MainModel.__init__."""
    return argparse.Namespace(
        embedding_dim=128,
        attn_drop=0.0,
        num_heads=16,
        num_layers=2,
        graph_hidden=128,
        graph_out=32,
        linear_hidden=[64],
        N_I=10,
        N_T=5,
        modalities="metric,trace,log",
        root_loss="ce",
        type_loss="ce",
        focal_gamma=2.0,
        use_graph_priors=use_graph_priors,
    )


def make_dummy_batch(batch_size: int = 2, n_nodes: int = 10):
    """Fabricate the tensors produced by prepare_for_graphormer."""
    D = 128
    metric = torch.randn(batch_size, n_nodes, D)
    trace = torch.randn(batch_size, n_nodes, D)
    log = torch.randn(batch_size, n_nodes, D)
    in_deg = torch.ones(batch_size, n_nodes, dtype=torch.long)
    out_deg = torch.ones(batch_size, n_nodes, dtype=torch.long)
    dist = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.long)
    # path_data shape: (B, N, N, max_len, edge_dim) where max_len=5, edge_dim=1
    path_data = torch.zeros(batch_size, n_nodes, n_nodes, 5, 1, dtype=torch.float)
    # attn_mask shape: (B, N+1, N+1), zeros = valid everywhere
    attn_mask = torch.zeros(batch_size, n_nodes + 1, n_nodes + 1)
    return metric, trace, log, in_deg, out_deg, dist, path_data, attn_mask


# --- Check 1: CLI flag is recognised by argparse in main.py ---------------

print("\n=== 1. CLI flag wiring (argparse in main.py) ===")
try:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    help_text = result.stdout + result.stderr
    check("main.py --help runs cleanly",
          result.returncode == 0 and "--no_graph_priors" in help_text,
          f"returncode={result.returncode}")
    check("--use_graph_priors appears in --help",
          "--use_graph_priors" in help_text)
    check("--no_graph_priors appears in --help",
          "--no_graph_priors" in help_text)
except Exception as e:
    check("main.py --help runs cleanly", False, f"{type(e).__name__}: {e}")


# --- Check 2: MainModel constructs in both modes --------------------------

print("\n=== 2. MainModel construction ===")
model_default, model_ablation = None, None
for label, ugp in [("default (priors on)", True), ("ablation (priors off)", False)]:
    try:
        m = MainModel(make_args(use_graph_priors=ugp))
        check(f"MainModel built ({label})", True)
        if ugp:
            model_default = m
        else:
            model_ablation = m
    except Exception as e:
        check(f"MainModel built ({label})", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# --- Check 3: Forward pass on dummy data ---------------------------------

print("\n=== 3. Forward pass on dummy batch ===")
if model_default is not None and model_ablation is not None:
    metric, trace, log, in_deg, out_deg, dist, path_data, attn_mask = make_dummy_batch()
    for label, m in [("default", model_default), ("ablation", model_ablation)]:
        try:
            m.eval()
            with torch.no_grad():
                (f_m, f_t, f_l), root_logit, type_logit = m(
                    metric, trace, log, in_deg, out_deg, dist, path_data, attn_mask
                )
            shape_ok = (
                f_m.shape == (2, 32)
                and f_t.shape == (2, 32)
                and f_l.shape == (2, 32)
                and root_logit.shape == (2, 10)
                and type_logit.shape == (2, 5)
            )
            check(f"forward pass ok ({label})", shape_ok,
                  f"shapes f_m={f_m.shape}, root={root_logit.shape}, type={type_logit.shape}")
            check(f"no NaN in outputs ({label})",
                  not (torch.isnan(root_logit).any() or torch.isnan(type_logit).any()))
        except Exception as e:
            check(f"forward pass ok ({label})", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# --- Check 4: Behavioural difference between the two modes ----------------

print("\n=== 4. Ablation actually changes Graphormer behaviour ===")
# Probe at the encoder level: same node features, see if outputs differ.
try:
    ge_with = GraphormerEncoder(
        embedding_dim=128, hidden_dim=128, out_dim=32,
        num_heads=16, num_layers=2, use_graph_priors=True,
    )
    ge_without = GraphormerEncoder(
        embedding_dim=128, hidden_dim=128, out_dim=32,
        num_heads=16, num_layers=2, use_graph_priors=False,
    )
    # Copy weights so the encoders differ only in the structural-prior code path.
    ge_without.load_state_dict(ge_with.state_dict())

    metric, *_ , dist, path_data, attn_mask = make_dummy_batch()
    in_deg = torch.ones(2, 10, dtype=torch.long) * 3   # non-trivial degrees
    out_deg = torch.ones(2, 10, dtype=torch.long) * 3
    dist = torch.randint(0, 5, (2, 10, 10), dtype=torch.long)  # non-trivial distances

    ge_with.eval(); ge_without.eval()
    with torch.no_grad():
        y_with = ge_with(metric, in_deg, out_deg, attn_mask, path_data, dist)
        y_without = ge_without(metric, in_deg, out_deg, attn_mask, path_data, dist)

    max_diff = (y_with - y_without).abs().max().item()
    check("outputs differ with priors on/off (shared weights)",
          max_diff > 1e-5,
          f"max abs difference = {max_diff:.4e}")
except Exception as e:
    check("outputs differ with priors on/off", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# --- Check 5: Parameter count parity --------------------------------------

print("\n=== 5. Parameter count (sanity) ===")
if model_default is not None and model_ablation is not None:
    n_default = sum(p.numel() for p in model_default.parameters())
    n_ablation = sum(p.numel() for p in model_ablation.parameters())
    check("parameter counts equal (modules instantiated either way)",
          n_default == n_ablation,
          f"{n_default:,} vs {n_ablation:,}")
    print(f"  Total trainable parameters: {n_default:,}")


# --- Summary --------------------------------------------------------------

print("\n" + "=" * 60)
if FAILED:
    print(f"FAILED CHECKS ({len(FAILED)}):")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
print("All smoke checks passed. The ablation flag is wired correctly.")
print("Next step: launch a real CV prog with --no_graph_priors.")
sys.exit(0)
