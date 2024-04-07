# Membership Inference Attacks to Graph Neural Networks for Graph Classification

This repository contains the source code corresponding to the ICDM2021 paper: "Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications". You can access the full version of the paper at [https://arxiv.org/abs/2110.08760](https://arxiv.org/abs/2110.08760)

Please cite the following paper if you use this code in your research:
<pre>@inproceedings{wypy2021miagnn, title={Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications}, author={Bang, Wu and Xiangwen, Yang and Shirui, Pan and Xingliang, Yuan}, booktitle={2021 IEEE International Conference on Data Mining (ICDM)}, year={2021}, organization={IEEE}}</pre>

# Installation

You may need to upgrade the **Lasagne** library if you encounter a version mismatch error. To upgrade, use the following command:
<pre>pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip</pre>

# Usage

## Step 1: Train the Target and Shadow Model

Use the following commands to train the victim GCN model.

For Dataset DD, PROTEINS_full, ENZYMES:
<pre>sh run_TUs_target_shadow_training.sh --number 10 --start_epoch 100 --dataset DD</pre>

And for SuperPixel Converted Graph Dataset CIFAR10, MNIST:
<pre>sh run_SPs_target_shadow_training.sh --number 10 --start_epoch 100 --dataset MNIST</pre>

The '--number' is the number of repeated model training. The '--start_epoch' is the minimum number of interactions to train the model. The '--dataset' is the dataset name of model training.

For detailed code execution, refer to 'main_SPs_graph_classification.py' and 'main_TUs_graph_classification.py' in t