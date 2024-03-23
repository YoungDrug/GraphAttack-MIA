import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
import os
from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
warnings.simplefilter("ignore")


def transfer_based_attack(epochs):
    # GCN_DD_GPU1_12h53m37s_on_Jan_28_2021 0.7900280269	0.6378787879
    # GCN_DD_GPU0_19h36m32s_on_Jan_27_2021  0.822117084 0.7315151515

    # GCN_PROTEINS_full_GPU0_03h11m51s_on_Jan_28_2021 0.7707677769	0.5766666667
    attack_base_path = 'data/statis/GCN/GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021/'
    target_base_path = 'data/statis/GCN/GCN_DD_GPU1_16h26m32s_on_Jun_08_2021/'
    # GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021 -> GCN_DD_GPU1_16h26m32s_on_Jun_08_2021
    # For attack dataset
    if os.listdir(attack_base_path).__contains__("S_RUN_"):
        S_X_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_0.pickle')
        S_Label_0_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_0.pickle')
        S_Label_1_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_1.pickle')
        S_Label_0_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_0.pickle')
        S_Label_1_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_1.pickle')
    else:
        S_X_train_in = load_pickled_data(attack_base_path + 'X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'y_train_Label_0.pickle')
    # For target Dataset
    if os.listdir(target_base_path).__contains__("T_RUN_"):
        T_X_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_0.pickle')
        T_Label_0_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_0.pickle')
        T_Label_1_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_1.pickle')
        T_Label_0_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_0.pickle')
        T_Label_1_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_1.pickle')
    else:
        T_X_train_in = load_pickled_data(target_base_path + 'X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'y_train_Label_0.pickle')

 