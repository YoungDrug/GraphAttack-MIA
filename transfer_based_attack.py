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
    target_base_path = 'data/s