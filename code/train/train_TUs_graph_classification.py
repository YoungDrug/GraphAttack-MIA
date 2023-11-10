
"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import pickle

import dgl
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from train.metrics import accuracy_TU as accuracy

"""
    For GCNs
"""


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        # print("batch_graphs:",batch_graphs)
        # print("batch_graph_size:",len(batch_graphs))
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        # print("batch_scores:",batch_scores)
        # print("batch_labels:",batch_labels)
        loss = model.loss(batch_scores, batch_labels)
        # print("loss:",loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    train_posterior = []
    train_labels = []
    num_nodes, num_edges = [],[]
    flag = []
    if type(epoch) is str:
        flag = epoch.split('|')
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # Calculate Posteriors
            if len(flag) == 3:
                graphs = dgl.unbatch(batch_graphs)
                for graph in graphs:
                    num_nodes.append(graph.number_of_nodes())
                    num_edges.append(graph.number_of_edges())
                for posterior in F.softmax(batch_scores, dim=1).detach().cpu().numpy().tolist():
                    train_posterior.append(posterior)
                    train_labels.append(int(flag[0]))

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()