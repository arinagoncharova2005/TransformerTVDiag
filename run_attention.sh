#!/usr/bin/env bash
# Attention extraction + aggregation runner.
#
# Usage: from TransTVDiag_experiments_results/, comment out lines you don't
# want, uncomment what you want, then `bash experimental/run_attention.sh`.
#
# Extraction MUST run on the server (DGL is broken locally).
# Aggregation is CPU-only and runs anywhere.
#
# Outputs:
#   logs/gaia/attention/{config}/fold_{k}.h5
#   logs/gaia/attention/{config}/fold_{k}_index.csv
#   logs/gaia/attention/{config}/aggregates/per_sample.csv
#   logs/gaia/attention/{config}/aggregates/modality_importance.csv
#   logs/gaia/attention/{config}/aggregates/{class}_{modality}_{layer0,last}.npy

set -e

# ============================================================================
# STEP A — extraction (server only)
# ============================================================================

# --- Smoke test on 64 samples (sanity check before a full run) -----
# python experimental/extract_attention.py --config cb_ce_awl --fold 0 --limit 64

# --- Single config × single fold (full fold) -----
# python experimental/extract_attention.py --config cb_ce_awl           --fold 0
# python experimental/extract_attention.py --config ce_awl              --fold 0
# python experimental/extract_attention.py --config focal_awl_gamma_2p0 --fold 0

# --- All 5 folds for one config (long; ~5x single-fold time) -----
# for fold in 0 1 2 3 4; do
#   python experimental/extract_attention.py --config cb_ce_awl --fold $fold
# done

# --- All 3 configs × fold 0 (the comparative claim) -----
# for cfg in cb_ce_awl ce_awl focal_awl_gamma_2p0; do
#     for fold in 0 1 2 3 4; do
#         python experimental/extract_attention.py --config $cfg --fold $fold
#     done
# done

# --- All 3 configs × all 5 folds (final run) -----
# for cfg in cb_ce_awl ce_awl focal_awl_gamma_2p0; do
#   for fold in 0 1 2 3 4; do
#     python experimental/extract_attention.py --config $cfg --fold $fold
#   done
# done

# ============================================================================
# STEP B — aggregation (anywhere; needs the .h5 + .csv from step A)
# ============================================================================

# --- One config, fold 0 only -----
# python experimental/aggregate_attention.py --config cb_ce_awl           --folds 0
# python experimental/aggregate_attention.py --config ce_awl              --folds 0
# python experimental/aggregate_attention.py --config focal_awl_gamma_2p0 --folds 0

# --- One config, all folds it finds (default --folds 0 1 2 3 4) -----
# python experimental/aggregate_attention.py --config cb_ce_awl
# python experimental/aggregate_attention.py --config ce_awl
# python experimental/aggregate_attention.py --config focal_awl_gamma_2p0

# --- All 3 configs in one go -----
# for cfg in cb_ce_awl ce_awl focal_awl_gamma_2p0; do
#   python experimental/aggregate_attention.py --config $cfg
# done

# ============================================================================
# DIAGNOSTICS — single-sample inspection (when aggregate numbers look off)
# ============================================================================

# python experimental/inspect_one_sample.py --config cb_ce_awl --fold 0 --sample-idx 0


for cfg in cb_ce_awl ce_awl focal_awl_gamma_2p0; do
  python experimental/aggregate_node_dependencies.py --config $cfg
done
