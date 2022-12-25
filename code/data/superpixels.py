import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import dgl
import torch
import torch.utils.data
import time


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_n