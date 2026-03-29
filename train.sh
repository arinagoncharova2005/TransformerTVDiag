#!/bin/bash

# configuration
# labels_file='labels_without_login.csv'
# labels_file='gaia.csv'
labels_file='label_15_85_corrected_index.csv'
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
root_loss='weighted_focal'
type_loss='weighted_focal'
focal_gamma=2.0

if [ "$dataset" = "gaia" ]; then
    python main.py --seed $seed --dataset $dataset --labels_file $labels_file  \
    --N_I 10 --N_T 5 --temperature 0.3  --epochs $epochs --lr $lr --batch_size $batch_size \
    --aggregator "lstm" --guide_weight $guide_weight --patience 5 --aug_percent $aug_percent \
    --root_loss $root_loss --type_loss $type_loss --focal_gamma $focal_gamma \
    --dynamic_weight --TO --CM --modalities metric,trace,log \
    --num_heads 16 --num_layers 2 --graph_hidden 128 --experiment_label "performance" \
    #--no_train --no_reconstruct \
    #--no_evaluate
fi