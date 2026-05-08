#!/bin/bash

# configuration
# labels_file='labels_without_login.csv'
labels_file='gaia.csv'
# labels_file='label_without_mobservice_correct_index.csv'
# labels_file='label_without_mobservice.csv'
# labels_file='gaia_without_mobservice.csv'
dataset='gaia'
# labels_file='label_15_85.csv'
epochs=3000
lr=0.001
batch_size=128
guide_weight=0.1
aug_percent=0.2
seed=42

# examples:
# root_loss='ce'
# root_loss='weighted_ce'
# root_loss='focal'
# root_loss='weighted_focal'
root_loss='focal'
type_loss='focal'
focal_gamma=0.5

if [ "$dataset" = "gaia" ]; then
    # python main.py --seed $seed --dataset $dataset --labels_file $labels_file  \
    # --N_I 10 --N_T 5 --temperature 0.3  --epochs $epochs --lr $lr --batch_size $batch_size \
    # --aggregator "lstm" --guide_weight $guide_weight --patience 5 --aug_percent $aug_percent \
    # --root_loss $root_loss --type_loss $type_loss --focal_gamma $focal_gamma \
    # --dynamic_weight --TO --CM --modalities metric,trace,log \
    # --num_heads 16 --num_layers 2 --graph_hidden 128 --experiment_label "performance" \
    # --model_filename "corrected_index_data_and_adaptive_weighting_trained_model_gaia_focal_2.pt"
    # # --no_train --no_reconstruct \
    # # --no_evaluate

    # Stratified 5-fold focal gamma sweep example:
    python main.py --seed $seed --dataset $dataset --labels_file $labels_file \
    --N_I 10 --N_T 5 --temperature 0.3 --epochs $epochs --lr $lr --batch_size $batch_size \
    --aggregator "lstm" --guide_weight $guide_weight --patience 5 --aug_percent $aug_percent \
    --root_loss $root_loss --type_loss $type_loss --focal_gamma_sweep "0.25" \
    --cv_folds 5 --cv_stratify_by anomaly_type \
    --dynamic_weight --TO --CM --modalities metric,trace,log \
    --num_heads 16 --num_layers 2 --graph_hidden 128 --experiment_label "cv-focal-sweep" \
    --eval_artifacts_dir "logs/gaia/cv_evaluation_artifacts"
    # -- model_filename "corrected_index_data_trained_model_gaia_focal_sweep_and.pt"
    # --focal_gamma_sweep "0.25,0.5,0.75,1.0,2.0"
fi