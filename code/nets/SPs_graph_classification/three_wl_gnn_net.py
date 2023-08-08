import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

"""
    3WLGNN / ThreeWLGNN
    Provably Powerful Graph Networks (Maron et al., 2019)
    https://papers.nips.cc/paper/8488-provably-powerful-graph-networks.pdf
    
    CODE adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/
"""

from layers.three_wl_gnn_layers import RegularBlock, MlpBlock, SkipConnection, FullyConnected, diag_offdiag_maxpool
from layers.mlp_readout_layer import MLPReadout

class ThreeWLGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim_node = net_params['in_dim']
        depth_of_mlp = net_params['depth_of_mlp']
        hidden_dim = net_params['hidden_dim