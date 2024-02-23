#!/bin/bash


############
# Usage
############

# bash script_main_superpixels_graph_classification_MNIST_100k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN



############
# MNIST - 4 RUNS  
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_superpixels_graph_classification.py 
tmux new -s benchmark -d
tmux send-keys "so