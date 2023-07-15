"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SPs_graph_classification.gated_gcn_net import GatedGCNNet
from nets.SPs_graph_classification.gcn_net import GCNNet
from nets.SPs_graph_classification.gat_net import GATNet
from nets.SPs_graph_classification.graphsage_net import GraphSageNet
from nets.SPs_graph_classification.gin_net import GINNet
from nets.SPs_graph_classification.mo_net import MoNet as MoNet_
from nets.SPs_graph_classification.mlp_net import MLPNet
from nets.SPs_graph_classification.ring_gnn_net import RingGNNNet
from nets.SPs_graph_classification.three_wl_gnn_net import ThreeWLGNNNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_par