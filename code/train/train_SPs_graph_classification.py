"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import os
import pickle

import dgl
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from train.metrics import accuracy_MNIST_CIFAR as accuracy

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels