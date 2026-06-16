#!/bin/bash
# Matched-N control experiment.
#
# Retrains the cb_ce + AWL model with every RCL instance capped to ~CAP train
# samples per fold (the majority instances mobservice1/2 are cut down to the
# rare size). Then reads the model's OWN per-instance RCL recall on the test
# set -- no probe, no embedding export needed.
#
# Reading the result: compare mobservice1/2 vs the rare instances at the SAME
# small train size.
#   - mob1/2 still high, rare still low  -> the gap is the signal, not N.
#   - mob1/2 drop to the rare level      -> the gap is sample size.
# (With balanced training the model loses its majority prior, so absolute
#  mob recall drops a bit on its own; read the mob-vs-rare GAP, not the number.)
#
# This varies the ENCODER training size, which the earlier probe-only control
# could not, so it speaks to the open question in the thesis.
#
# Heavy: full encoder retraining x5 folds on GPU. Tiny data per class -> noisy,
# consider repeating with a few seeds (change --seed) and averaging.

set -e

CAP=50
LABEL="cb_ce_awl_cap${CAP}"
ARTIFACTS="logs/gaia/cv_evaluation_artifacts/${LABEL}"

# 1. Retrain with all classes capped to CAP train samples/fold, and write the
#    per-fold test predictions + per-instance RCL classification reports.
python main.py --seed 42 --dataset gaia --labels_file gaia.csv \
  --N_I 10 --N_T 5 --temperature 0.3 --epochs 3000 --lr 0.001 --batch_size 128 \
  --aggregator lstm --guide_weight 0.1 --patience 5 --aug_percent 0.2 \
  --root_loss cb_ce --type_loss cb_ce --focal_gamma_sweep "2.0" --cb_beta 0.999 \
  --cv_folds 5 --cv_stratify_by instance \
  --dynamic_weight --TO --CM --modalities metric,trace,log \
  --num_heads 16 --num_layers 2 --graph_hidden 128 --graph_out 32 --embedding_dim 128 \
  --cap_per_instance ${CAP} \
  --experiment_label "${LABEL}" \
  --eval_artifacts_dir "${ARTIFACTS}"

# 2. Pool the per-fold predictions and print per-instance RCL recall.
python experimental/summarize_matched_n.py --artifacts_dir "${ARTIFACTS}"

echo
echo "Done. Per-fold reports in ${ARTIFACTS} (*_classification_report_rcl.csv)."
echo "Compare mobservice1/2 recall vs the rare instances now that all are ~${CAP}/fold."
