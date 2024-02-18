#!/bin/bash


############
# Usage
############

# bash script_main_superpixels_graph_classification_CIFAR10_500k.sh



############
# GNNs
############


#3WLGNN
#RingGNN



############
# CIFAR10 - 4 RUNS  
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_superpixels_graph_classification.py 
tmux new -s benchmark_CIFAR10 -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=CIFAR10
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
python $code --dataset $dataset