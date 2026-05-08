#!/bin/bash

# Common configuration
labels_file='gaia.csv'
dataset='gaia'
epochs=3000
lr=0.001
batch_size=128
guide_weight=0.1
aug_percent=0.2
seed=42

# Common flags for all experiments
common="--seed $seed --dataset $dataset --labels_file $labels_file \
    --N_I 10 --N_T 5 --temperature 0.3 --epochs $epochs --lr $lr --batch_size $batch_size \
    --aggregator lstm --guide_weight $guide_weight --patience 5 --aug_percent $aug_percent \
    --TO --CM --modalities metric,trace,log \
    --num_heads 16 --num_layers 2 --graph_hidden 128 \
    --cv_folds 5 --cv_stratify_by instance"

# =============================================
# Uncomment ONE experiment block and run
# =============================================
# 1 2 6 запущены
# 3 надо с разными параметрами (1, 2)

# --- 1. CE + AWL ---
# python main.py $common --root_loss ce --type_loss ce \
#     --dynamic_weight --experiment_label "ce_awl"

# --- 2. Weighted CE + AWL ---
# python main.py $common --root_loss weighted_ce --type_loss weighted_ce \
#     --dynamic_weight --experiment_label "weighted_ce_awl"

# # --- 3. Focal + AWL (with AWL clamp fix) ---
# python main.py $common --root_loss focal --type_loss focal \
#     --focal_gamma 2 --dynamic_weight --experiment_label "focal_awl"

## focal gamma 1 results
# 2026-04-16 22:31:57 [INFO]: CV per-fold results saved to logs/gaia/cv_results_2026-04-16T16-49-15.263919.csv
# 2026-04-16 22:31:57 [INFO]: CV summary saved to logs/gaia/cv_results_2026-04-16T16-49-15.263919_summary.csv

# --- 4. DRW: CE warmup -> Focal + AWL ---
# python main.py $common --root_loss focal --type_loss focal \
#     --focal_gamma 2 --dynamic_weight --drw --experiment_label "drw_focal_awl_focal_gamma_2"

# # --- 5. Class-Balanced Focal + AWL ---
python main.py $common --root_loss cb_focal --type_loss cb_focal \
    --focal_gamma 2 --cb_beta 0.999 --dynamic_weight --experiment_label "cb_focal_awl_focal_gamma_2"

# --- 6. Class-Balanced CE + AWL ---
# python main.py $common --root_loss cb_ce --type_loss cb_ce \
#     --cb_beta 0.999 --dynamic_weight --experiment_label "cb_ce_awl"

# Артефакты 6
# 2026-04-16 16:09:42 [INFO]: Evaluated checkpoint: logs/gaia/cv_checkpoints/cb_ce_awl/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4.pt
# 2026-04-16 16:09:42 [INFO]: The average test time is 0.007541047243084741[s]
# 2026-04-16 16:09:42 [INFO]: The total test time is 24.440534114837646[s]
# 2026-04-16 16:09:42 [INFO]: Evaluation predictions saved to logs/gaia/evaluation_artifacts/cb_ce_awl-gamma_2.0-fold_4/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4_test_predictions.csv
# 2026-04-16 16:09:42 [INFO]: RCL classification report saved to logs/gaia/evaluation_artifacts/cb_ce_awl-gamma_2.0-fold_4/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4_classification_report_rcl.csv
# 2026-04-16 16:09:42 [INFO]: FTI classification report saved to logs/gaia/evaluation_artifacts/cb_ce_awl-gamma_2.0-fold_4/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4_classification_report_fti.csv
# 2026-04-16 16:09:42 [INFO]: RCL confusion matrix saved to logs/gaia/evaluation_artifacts/cb_ce_awl-gamma_2.0-fold_4/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4_confusion_matrix_rcl.csv
# 2026-04-16 16:09:42 [INFO]: FTI confusion matrix saved to logs/gaia/evaluation_artifacts/cb_ce_awl-gamma_2.0-fold_4/TransTVDiag-root_cb_ce-type_cb_ce-gamma_2p0-fold_4_confusion_matrix_fti.csv
# 2026-04-16 16:09:42 [INFO]: CV per-fold results saved to logs/gaia/cv_results_2026-04-16T10-55-28.836891.csv
# 2026-04-16 16:09:42 [INFO]: CV summary saved to logs/gaia/cv_results_2026-04-16T10-55-28.836891_summary.csv