# %% [markdown]
# # Targeted Selection Demo For Caltech-101 Datasets With Rare Classes

# %% [markdown]
# ### Imports

# %%
import time
import random
import datetime
import copy
import numpy as np
from tabulate import tabulate
import os
import csv
import json
import subprocess
import sys
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from matplotlib import pyplot as plt
import sys
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from trust.utils.paths import RESULTS_DIR
# from trust.utils.models.resnet_caltech import ResNet18Caltech as ResNet18
# from trust.utils.models.resnet_caltech import ResNet50Caltech as ResNet50
from trust.utils.models.resnet import ResNet18
from trust.utils.models.resnet import ResNet50
from torch.utils.data import Subset, Dataset
from torch.autograd import Variable
import tqdm
from math import floor
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
# from trust.strategies.smi import SMI
# from trust.strategies.scmi import SCMI
from trust.strategies.random_sampling import RandomSampling
from trust.strategies.wassal_multiclass import WASSAL_Multiclass
from trust.strategies.modern_al import TypiClust, ProbCover, DCoM, ALFAMargin
from trust.strategies.sota_al import MaxHerding, UHerding, WassersteinIP
# from trust.strategies.wassal_private import WASSAL_P

from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.glister import GLISTER
from distil.active_learning_strategies.gradmatch_active import GradMatchActive
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.least_confidence_sampling import (
    LeastConfidenceSampling,
)
from distil.active_learning_strategies.margin_sampling import MarginSampling

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from trust.utils.utils import *
from trust.utils.viz import tsne_smi
import math
from random import shuffle

# Toggle: when True, adds Eq 7's soft-loss weighted training, which this
# driver originally never implemented at all (WASSAL_WITHSOFT behaved
# identically to plain WASSAL). When False, keeps that original behavior.
PAPER_ALIGNED_SOFT_LOSS = True

# %% [markdown]
# ### Helper functions


# %%
def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def create_model(name, num_cls, device, embedding_type):
    if name == "ResNet18":
        if embedding_type == "gradients":
            model = ResNet18(num_cls)
        else:
            model = ResNet18(num_cls)
    elif name == "ResNet50":
        if embedding_type == "gradients":
            model = ResNet50(num_cls)
        else:
            model = ResNet50(num_cls)
    elif name == "MnistNet":
        model = MnistNet()
    elif name == "ResNet164":
        model = ResNet164(num_cls)
    model.apply(init_weights)
    model = model.to(device)
    return model


def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    return criterion, criterion_nored


def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=m, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
    #     optimizer = optim.Adam(model.parameters(),weight_decay=wd)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=m, weight_decay=wd
    )
    return optimizer


def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing / 3600


def displayTable(val_err_log, tst_err_log):
    col1 = [str(i) for i in range(num_cls)]
    val_acc = [str(100 - i) for i in val_err_log]
    tst_acc = [str(100 - i) for i in tst_err_log]
    table = [col1, val_acc, tst_acc]
    table = map(list, zip(*table))
    print(
        tabulate(
            table, headers=["Class", "Val Accuracy", "Test Accuracy"], tablefmt="orgtbl"
        )
    )


def find_err_per_class(
    test_set,
    val_set,
    final_val_classifications,
    final_val_predictions,
    final_tst_classifications,
    final_tst_predictions,
    saveDir,
    prefix,
    doIdisplayTable=True,
):
    val_err_idx = list(np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        tst_class_idxs = list(
            torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy()
        )
        val_class_idxs = list(
            torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy()
        )
        # err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if len(val_class_idxs) > 0:
            val_error_perc = round(
                (len(val_err_class_idx) / len(val_class_idxs)) * 100, 2
            )
        else:
            val_error_perc = 0
        tst_error_perc = round((len(tst_err_class_idx) / len(tst_class_idxs)) * 100, 2)
        #         print("val, test error% for class ", i, " : ", val_error_perc, tst_error_perc)
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)
    if doIdisplayTable:
        displayTable(val_err_log, tst_err_log)
    tst_err_log.append(sum(tst_err_log) / len(tst_err_log))
    val_err_log.append(sum(val_err_log) / len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs


def aug_train_subset(
    train_set,
    lake_set,
    true_lake_set,
    subset,
    lake_subset_idxs,
    budget,
    augrandom=False,
):
    
    all_lake_idx = list(range(len(lake_set)))
    if budget >= len(lake_set):
        print(f"Budget ({budget}) >= lake set size ({len(lake_set)}). Using all remaining lake samples.")
        subset = all_lake_idx
        lake_subset_idxs = all_lake_idx
    elif not (len(subset) == budget) and augrandom:
        print(
            "Budget not filled, adding ", str(int(budget) - len(subset)), " randomly."
        )
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        # NEW: Don't try to sample more than available
        remain_budget = min(remain_budget, len(remain_lake_idx))
        if remain_budget > 0:
            random_subset_idx = list(
                np.random.choice(
                    np.array(remain_lake_idx), size=int(remain_budget), replace=False
                )
            )
            subset += random_subset_idx
    if not (len(subset) == budget) and augrandom:
        print(
            "Budget not filled, adding ", str(int(budget) - len(subset)), " randomly."
        )
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(
            np.random.choice(
                np.array(remain_lake_idx), size=int(remain_budget), replace=False
            )
        )
        subset += random_subset_idx
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        lake_ss = SubsetWithTargets(
            true_lake_set,
            subset,
            torch.Tensor(true_lake_set.targets.astype(np.float))[subset],
        )
    else:
        lake_ss = SubsetWithTargets(
            true_lake_set, subset, torch.Tensor(true_lake_set.targets.float())[subset]
        )
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_lake_set = SubsetWithTargets(
            lake_set,  # ← Changed from true_lake_set
            remain_lake_idx,
            torch.Tensor(lake_set.targets.astype(np.float))[remain_lake_idx],
        )
    else:
        remain_lake_set = SubsetWithTargets(
            lake_set,  # ← Changed from true_lake_set
            remain_lake_idx,
            torch.Tensor(lake_set.targets.float())[remain_lake_idx],
        )
    
    # ADD THIS NEW BLOCK (missing in your code):
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_true_lake_set = SubsetWithTargets(
            true_lake_set,
            remain_lake_idx,
            torch.Tensor(true_lake_set.targets.astype(np.float))[remain_lake_idx],
        )
    else:
        remain_true_lake_set = SubsetWithTargets(
            true_lake_set,
            remain_lake_idx,
            torch.Tensor(true_lake_set.targets.float())[remain_lake_idx],
        )
    
    aug_train_set = ConcatWithTargets(train_set, lake_ss)
    
    # CHANGE THIS LINE:
    return aug_train_set, remain_lake_set, remain_true_lake_set, lake_ss


def getPerClassSel(dset, sel_idxs, num_cls):
    perClsSel = [0 for i in range(num_cls)]
    if str(type(dset.targets)) == "<class 'numpy.ndarray'>":
        for idx in sel_idxs:
            perClsSel[int(dset.targets[idx])] += 1
    else:
        for idx in sel_idxs:
            perClsSel[int(dset.targets[idx].item())] += 1
    return perClsSel


def getQuerySet(train_set, query_set_targets, recipe="vanilla"):
    if recipe == "asis":
        return train_set
    train_idxs = []
    if str(type(train_set.targets)) == "<class 'numpy.ndarray'>":
        for idx, target in enumerate(train_set.targets):
            if target in query_set_targets:
                train_idxs.append(idx)
    else:
        for idx, target in enumerate(train_set.targets):
            if target.item() in query_set_targets:
                train_idxs.append(idx)
    if str(type(train_set.targets)) == "<class 'numpy.ndarray'>":
        query_dataset = SubsetWithTargets(
            train_set,
            train_idxs,
            torch.Tensor(train_set.targets.astype(np.float))[train_idxs],
        )
    else:
        query_dataset = SubsetWithTargets(
            train_set, train_idxs, torch.Tensor(train_set.targets.float())[train_idxs]
        )
    return query_dataset


def getPrivateSet(train_set, private_set_targets):
    train_idxs = []
    if str(type(train_set.targets)) == "<class 'numpy.ndarray'>":
        for idx, target in enumerate(train_set.targets):
            if target not in private_set_targets:
                train_idxs.append(idx)
    else:
        for idx, target in enumerate(train_set.targets):
            if target.item() not in private_set_targets:
                train_idxs.append(idx)
    if str(type(train_set.targets)) == "<class 'numpy.ndarray'>":
        private_dataset = SubsetWithTargets(
            train_set,
            train_idxs,
            torch.Tensor(train_set.targets.astype(np.float))[train_idxs],
        )
    else:
        private_dataset = SubsetWithTargets(
            train_set, train_idxs, torch.Tensor(train_set.targets.float())[train_idxs]
        )
    return private_dataset


def plotsimpelxDistribution(lake_set, classwise_final_indices_simplex,folder_name):
    simplex_values_dict = {}
    data_to_store = []

    for (
        selected_indices,
        simplex_query,
        class_idx,
    ) in classwise_final_indices_simplex:
        # Create a histogram of the simplex query values
        # plt.figure(figsize=(10, 5))
        # plt.hist(simplex_query.numpy(), bins=50, alpha=0.7)
        # plt.title(
        #     "Distributions of the simplex query for hypothesised Class: "
        #     + str(class_idx)
        # )
        # plt.xlabel("Query values")
        # plt.ylabel("Frequency")
        # plt.savefig(os.path.join(folder_name,"cifar10_simplex_distribution_class_{}.png".format(class_idx)))
        # plt.close()

        # # Create a histogram of the simplex query values with the targets represented as bars
        # unique_targets = np.unique(lake_set.targets)

        # # Create bins for the histogram
        # bin_edges = np.linspace(
        #     simplex_query.min().item(), simplex_query.max().item(), 50
        # )

        # # Choose a colormap
        # cmap = plt.cm.get_cmap("tab10")
        # colors = [cmap(i) for i in range(len(unique_targets))]

        # # Create a dictionary to map target to color
        # color_map = {target: color for target, color in zip(unique_targets, colors)}

        # # Prepare bin data to color based on target
        # bin_data = {target: [] for target in unique_targets}

        # for i in range(len(bin_edges) - 1):
        #     bin_mask = (simplex_query >= bin_edges[i]) & (
        #         simplex_query < bin_edges[i + 1]
        #     )
        #     bin_targets = np.array(lake_set.targets)[bin_mask]
        #     for target in unique_targets:
        #         count_target = np.sum(bin_targets == target)
        #         # Add the count to the bin_data if it's less than or equal to 100
        #         if count_target <= 200:
        #             bin_data[target].append(count_target)
        #         else:
        #             bin_data[target].append(0)

        # # Plot
        # plt.figure(figsize=(10, 5))
        # bottom = np.zeros(len(bin_edges) - 1)
        # for target, counts in bin_data.items():
        #     plt.bar(
        #         bin_edges[:-1],
        #         counts,
        #         width=np.diff(bin_edges),
        #         align="edge",
        #         label=str(target),
        #         bottom=bottom,
        #         color=color_map[target],
        #     )
        #     bottom += counts

        # plt.title(
        #     "Distributions of the simplex query for hypothesised Class: "
        #     + str(class_idx)
        # )
        # plt.xlabel("Query values")
        # plt.ylabel("Frequency")
        # plt.legend(title="Targets", bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig(os.path.join(folder_name,"caltech101_simplex_distribution_class_{}.png".format(class_idx)))
        # plt.close()
        
        # Update the simplex values dictionary
        simplex_values_dict[class_idx] = simplex_query.numpy()
    num_classes = 102  # Caltech-101 has 102 classes
    for idx, real_class in enumerate(lake_set.targets):
        #real class is stored as tensor(0). Make it into just label ie number

        values = [simplex_values_dict[i][idx] for i in range(num_classes)]
        if any(value != 0 for value in values):
            data_to_store.append((real_class.item(), *values))

    # Serialize and save the data to CSV
    data_file_path = os.path.join(folder_name, "simplex_data.csv")
    with open(data_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerows(data_to_store)



def print_final_results(res_dict, sel_cls_idx):
    print(
        "Gain in overall test accuracy: ",
        res_dict["test_acc"][1] - res_dict["test_acc"][0],
    )
    # bf_sel_cls_acc = np.array(res_dict['all_class_acc'][0])[sel_cls_idx]
    # af_sel_cls_acc = np.array(res_dict['all_class_acc'][1])[sel_cls_idx]
    # print("Gain in targeted test accuracy: ", np.mean(af_sel_cls_acc-bf_sel_cls_acc))


def analyze_simplex(args, unlabeled_set, simplex_query):
    print("======== analysis on simplex =========")
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_set, batch_size=len(unlabeled_set), shuffle=False
    )
    u_imgs, u_labels = next(iter(unlabeled_loader))
    u_imgs, u_labels = u_imgs.to(args["device"]), u_labels.to(args["device"])
    nz_query_idx = simplex_query.nonzero()

    # Using a loop to accommodate an array of target values
    total_correctly_identified = 0
    for query_value in args["target"]:
        num_nz_query = (u_labels[nz_query_idx] == query_value).nonzero().shape[0]
        total_correctly_identified += num_nz_query
    print(
        "no of query labels identified correctly: {}/{}".format(
            total_correctly_identified, nz_query_idx.shape[0]
        )
    )

    total_query_weight = 0
    for query_value in args["target"]:
        query_idx = torch.where(u_labels == query_value)
        query_weight = torch.sum(simplex_query[query_idx])
        total_query_weight += query_weight
    print("Weight of Query samples in simplex_query: {}".format(total_query_weight))


class WeightedDataset(Dataset):
    def __init__(self, imgs, targets, simplex_query, private_targets, simplex_private):
        self.imgs = imgs
        self.targets = targets
        self.simplex_query = simplex_query
        self.private_targets = private_targets
        self.simplex_private = simplex_private

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.imgs[idx]
        target = self.targets[idx]
        t = self.simplex_query[idx].item()
        private_target = (
            self.private_targets[idx] if self.private_targets is not None else []
        )
        p = self.simplex_private[idx].item() if self.simplex_private is not None else []

        return (image, target, t, private_target, p)


# return the elements from the simplex_query that contribute to the given percentage
def top_elements_contribute_to_percentage(simplex_query, n_percent, budget):
    # Pair each value with its original index
    indexed_simplex = list(enumerate(simplex_query))

    # Sort based on the value (in descending order)
    sorted_simplex = sorted(indexed_simplex, key=lambda x: x[1], reverse=True)

    # Calculate the total sum of the array
    total_sum = sum(value for index, value in sorted_simplex)

    # If the array doesn't sum up to 1, you might want to handle this case
   # if total_sum != 1:
   #     print("Total sum of simplex is", total_sum)

    target_sum = n_percent / 100.0  # Convert percentage to fraction
    cumulative_sum = 0
    selected_indices = []

    # Iterate over the sorted array
    for i, (index, value) in enumerate(sorted_simplex):
        cumulative_sum += value
        selected_indices.append(index)
        if cumulative_sum >= target_sum:
            break

    # Return values and their original indices
    selected_values = [simplex_query[i] for i in selected_indices]
    # if(len(selected_values)>budget) return only the top budget elements:
    if len(selected_values) > budget:
        selected_values = selected_values[:budget]
        selected_indices = selected_indices[:budget]

    return selected_values, selected_indices


# %% [markdown]
# # Custom Dataset Loader for Caltech-101

# %%
class Caltech101Dataset(Dataset):
    """Custom Dataset for Caltech-101 loaded from .npz files"""
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of shape (N, C, H, W) - images in [0,1] range
            y: numpy array of labels (N,)
            transform: optional transforms
        """
        self.data = torch.FloatTensor(X)  # Already in [0,1] range
        self.targets = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_caltech101_custom(datadir, feature, split_cfg):
    """
    Load Caltech-101 from .npz files and create custom splits
    Similar to load_dataset_custom for CIFAR-10
    
    Args:
        datadir: directory containing .npz files
        feature: 'classimb' for class imbalance experiments
        split_cfg: dictionary with split configuration
        
    Returns:
        train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls
    """
    print("Loading Caltech-101 from .npz files...")
    
    # Load train data (90% split)
    train_data = np.load(os.path.join(datadir, 'caltech101_train_90_32.npz'))
    X_train_full, y_train_full = train_data['X'], train_data['y']
    
    # Load test data (10% split)
    test_data = np.load(os.path.join(datadir, 'caltech101_test_10_32.npz'))
    X_test_full, y_test_full = test_data['X'], test_data['y']
    
    print(f"Loaded train: {X_train_full.shape}, test: {X_test_full.shape}")
    print(f"Number of classes: {len(np.unique(y_train_full))}")
    
    # Get class distribution
    unique_classes, train_counts = np.unique(y_train_full, return_counts=True)
    num_cls = len(unique_classes)
    
    if feature == "classimb":
        # Class imbalance scenario - create custom splits
        per_class_train = split_cfg["per_class_train"]
        per_class_val = split_cfg["per_class_val"]
        per_class_lake = split_cfg["per_class_lake"]
        sel_cls_idx = split_cfg["sel_cls_idx"]
        
        train_indices = []
        val_indices = []
        lake_indices = []
        
        for cls_idx in range(num_cls):
            # Get all indices for this class
            cls_mask = (y_train_full == cls_idx)
            cls_indices = np.where(cls_mask)[0]
            
            # Shuffle indices for this class
            np.random.shuffle(cls_indices)
            
            # Determine split sizes for this class
            n_train = min(per_class_train[cls_idx], len(cls_indices))
            n_val = min(per_class_val[cls_idx], len(cls_indices) - n_train)
            n_lake = min(per_class_lake[cls_idx], len(cls_indices) - n_train - n_val)
            
            # Split indices
            train_indices.extend(cls_indices[:n_train])
            val_indices.extend(cls_indices[n_train:n_train + n_val])
            lake_indices.extend(cls_indices[n_train + n_val:n_train + n_val + n_lake])
        
        # Create datasets
        train_set = Caltech101Dataset(X_train_full[train_indices], y_train_full[train_indices])
        val_set = Caltech101Dataset(X_train_full[val_indices], y_train_full[val_indices])
        lake_set = Caltech101Dataset(X_train_full[lake_indices], y_train_full[lake_indices])
        test_set = Caltech101Dataset(X_test_full, y_test_full)
        
        print(f"Split sizes - Train: {len(train_set)}, Val: {len(val_set)}, Lake: {len(lake_set)}, Test: {len(test_set)}")
        
    else:
        # Standard split - use all data
        train_set = Caltech101Dataset(X_train_full, y_train_full)
        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])
        
        lake_set = train_set  # Use training set as lake
        test_set = Caltech101Dataset(X_test_full, y_test_full)
        sel_cls_idx = list(range(num_cls))
    
    return train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls


# %% [markdown]
# # Data, Model & Experimental Settings
# The Caltech-101 dataset contains ~9,000 images across 101 object categories (plus background = 102 classes).
# Images are resized to 224×224. We simulate a class imbalance scenario using the split_cfg dictionary.
# We use a ResNet18 model as our task DNN and train it on the simulated imbalanced version of Caltech-101.
# We perform targeted selection using various active learning strategies.

# %%
feature = "classimb"

# datadir = 'data/'
datadir = "data"  # contains the npz file of the caltech101 dataset

data_name = "caltech101"

learning_rate = 0.0003
computeClassErrorLog = True
if __name__ == "__main__":
    # Accept skip_strategies and skip_budgets from command line arguments
    experiment_name=sys.argv[5]
    device_id = int(sys.argv[4])
    print('setting deviceid to',str(device_id))
else:
    device_id=1
    print('setting deviceid to default',str(device_id))
device = "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
miscls = False  # Set to True if only the misclassified examples from the imbalanced classes is to be used

num_cls = 102  # Caltech-101 has 102 classes (101 objects + 1 background)
# budget = 10
visualize_tsne = False

# Split configuration for Caltech-101
# Adjusted for smaller dataset size and class imbalance
split_cfg = {
    "num_cls_imbalance": list(range(102)),
    "sel_cls_idx": list(range(102)),
    "per_class_train": [8] * 102,  # 10% of train data
    "per_class_val": [5] * 102,     # 6% of train data
    "per_class_lake": [45] * 102,    #  (~60% of train data)
    "per_imbclass_train": [8] * 102,
    "per_imbclass_val": [5] * 102,
    "per_imbclass_lake": [45] * 102,
}

print("split_cfg:", split_cfg)

# %% [markdown]
# # Targeted Selection Algorithm
# 1. Given: Initial Labeled set of Examples: 𝐸, large unlabeled dataset: 𝑈, A target subset/slice where we want to improve accuracy: 𝑇, Loss function 𝐿 for learning
# 2. Train model with loss $\mathcal L$ on labeled set $E$ and obtain parameters $\theta_E$
# 3. Compute the gradients $\{\nabla_{\theta_E} \mathcal L(x_i, y_i), i \in U\}$ (using hypothesized labels) and $\{\nabla_{\theta_E} \mathcal L(x_i, y_i), i \in T\}$.
# (This notebook uses gradients for representation. However, any other representation can be used. Trust also supports using features via the API.)
# 4. Compute the similarity kernels $S$ (this includes kernel of the elements within $U$, within $T$ and between $U$ and $T$) and define a submodular function $f$ and diversity function $g$
# 5. Compute subset $\hat{A}$ by mazximizing the SMI function: $\hat{A} \gets \max_{A \subseteq U, |A|\leq k} I_f(A;T) + \gamma g(A)$
# 6. Obtain the labels of the elements in $A^*$: $L(\hat{A})$
# 7. Train a model on the combined labeled set $E \cup L(\hat{A})$


# %%
def run_targeted_selection(
    dataset_name,
    datadir,
    feature,
    model_name,
    budget,
    split_cfg,
    learning_rate,
    run,
    device,
    computeErrorLog,
    strategy="SIM",
    sf="",
    embedding_type="features",
    soft_loss_hyperparam="3"
):
    # load the dataset in the class imbalance setting
    train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_caltech101_custom(
        datadir, feature, split_cfg
    )
    print("Indices of randomly selected classes for imbalance: ", sel_cls_idx)

    # Set batch size for train, validation and test datasets
    # Reduced batch sizes for 224x224 images (much larger than 32x32 CIFAR-10)
    N = len(train_set)
    trn_batch_size = 1024  # Reduced from 1000
    val_batch_size = 256  # Reduced from 200
    tst_batch_size = 256 # Reduced from 200

    # # Create dataloaders
    # trainloader = torch.utils.data.DataLoader(
    #     train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True
    # )

    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True
    )

    tstloader = torch.utils.data.DataLoader(
        test_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True
    )

    # lakeloader = torch.utils.data.DataLoader(
    #     lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True
    # )
    true_lake_set = copy.deepcopy(lake_set)
    # Budget for subset selection
    bud = budget
    # soft subset max budget % of the lake set
    ss_max_budget_percentage = 80
    # Variables to store accuracies
    num_rounds = 8  # The first round is for training the initial model and the second round is to train the final model
    fulltrn_losses = np.zeros(num_rounds)
    trn_losses = np.zeros(num_rounds)
    val_losses = np.zeros(num_rounds)
    tst_losses = np.zeros(num_rounds)
    subtrn_losses = np.zeros(num_rounds)
    timing = np.zeros(num_rounds)
    trn_acc = np.zeros(num_rounds)
    val_acc = np.zeros(num_rounds)
    tst_acc = np.zeros(num_rounds)
    full_trn_acc = np.zeros(num_rounds)
    csvlog = []
    val_csvlog = []
    # Results dictionary
    res_dict = {
        "test_acc": [],
        "sel_per_cls": [],
        "all_class_acc": [],
        "all_val_class_acc": [],
    }

    # Create directories for logging
    all_logs_dir = (
        str(RESULTS_DIR) + "/"
        + experiment_name
        + "/"
        + dataset_name
        + "/"
        + feature
        + "/rounds"
        + str(num_rounds)
        + "/"
        + sf
        + "/"
        + str(bud)
        + "/"
        + str(run)
    )
    subprocess.run(["mkdir", "-p", all_logs_dir])
    exp_name = "results_" + sf + "_" + str(bud)

    # Resume support: this (experiment, budget, strategy) cell's result
    # is only ever written once, at the very end of this function, after
    # all num_rounds AL rounds complete. There was previously no way to
    # skip a cell whose result already exists short of manually computing
    # skip_strategies/skip_budgets CLI arguments (coarse: applies across
    # every experiment). If this exact cell already has a saved result,
    # skip it before any GPU work begins - this makes relaunching after
    # an interruption (e.g. an OOM from unrelated GPU contention) resume
    # from wherever it left off, rather than repeating completed cells.
    # A cell interrupted mid-way (no result JSON yet) is NOT resumed at
    # the round level - it restarts from round 0 when relaunched.
    result_json_path = os.path.join(all_logs_dir, exp_name + ".json")
    if os.path.exists(result_json_path):
        print(
            "Skipping "
            + exp_name
            + " (run "
            + str(run)
            + "): result already exists at "
            + result_json_path
        )
        return

    # Model, optimizer, loss function
    model = create_model(model_name, num_cls, device, embedding_type)
    criterion, criterion_nored = loss_function()

    # Strategy arguments
    strategy_args = {
        "batch_size": 1024,  # Reduced batch size for larger images
        "device": device,
        # "loss": criterion_nored,
        "model": model,
        "embedding_type": embedding_type,
        "keep_embedding": True,
        "lr": 0.001,
        "wassal_iterations": 10,  # ← This stops at epoch 9
        "step_size": 10,
        "min_iteration": 5,
    }
    if "WITHSOFT" in strategy or strategy == "WASSAL":
        strategy_args_softsubset = {
            "batch_size": 1024,  # Reduced batch size
            "device": device,
            "loss": criterion_nored,
            "model": model,
            "embedding_type": embedding_type,
            "keep_embedding": True,
            "lr": 0.001,
            "wassal_iterations": 10,  # ← This stops at epoch 9
            "step_size": 10,
            "min_iteration": 5,
        }
        # Calculate soft subset budget (per class). SVHN/CIFAR-10/Pneumonia
        # use small fixed per-class caps (100/400/500) that keep the total
        # soft-subset in the low thousands across their ~2-10 classes. The
        # previous formula here (80% of the whole lake_set, per class) was
        # never actually exercised before this mechanism was wired up: with
        # 102 classes it produced ~42k soft-subset entries from a ~3.6k-image
        # lake_set (11x+ duplication), wildly disproportionate to the other
        # datasets. Use a small per-class cap instead, scaled down for the
        # much larger class count so the total stays in the same ballpark.
        ss_budget = 30
        if ss_budget > len(lake_set):
            ss_budget = len(lake_set)
        strategy_args_softsubset["soft_loss_hyperparam"] = soft_loss_hyperparam

    # Initialize strategy for soft subset if needed
    if "WITHSOFT" in strategy or strategy == "WASSAL":
        print("Initializing WASSAL/soft subset strategy")
        unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
        for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
        strategy_softsubset = WASSAL_Multiclass(
            train_set,
            unlabeled_lake_set,
            for_query_set,
            model,
            num_cls,
            strategy_args_softsubset,
        )

    # Initialize the main strategy
    unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
    if strategy == "AL" or strategy == "AL_WITHSOFT":
        if sf == "glister" or sf == "glister_withsoft":
            strategy_sel = GLISTER(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set
            )
        elif sf == "us" or sf == "us_withsoft":
            strategy_sel = EntropySampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "badge" or sf == "badge_withsoft":
            strategy_sel = BADGE(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "glister-tss":
            strategy_sel = GLISTER(
                train_set,
                unlabeled_lake_set,
                model,
                num_cls,
                strategy_args,
                val_set,
                typeOf="rand",
                lam=0.1,
            )
        elif sf == "gradmatch-tss" or sf == "gradmatch-tss_withsoft":
            strategy_sel = GradMatchActive(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set
            )
        elif sf == "coreset" or sf == "coreset_withsoft":
            strategy_sel = CoreSet(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "leastconf" or sf == "leastconf_withsoft":
            strategy_sel = LeastConfidenceSampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "margin" or sf == "margin_withsoft":
            strategy_sel = MarginSampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "typiclust":
            strategy_sel = TypiClust(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "probcover":
            strategy_sel = ProbCover(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "dcom":
            strategy_sel = DCoM(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "alfamargin":
            strategy_sel = ALFAMargin(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "maxherding":
            strategy_sel = MaxHerding(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "uherding":
            strategy_sel = UHerding(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "wassersteinip":
            strategy_sel = WassersteinIP(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )

    elif strategy == "SIM" or strategy == "SIM_WITHSOFT":
        strategy_args["smi_function"] = sf
        strategy_args["optimizer"] = "LazyGreedy"
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        strategy_sel = SMI(
            train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args
        )

    elif strategy == "SCMI" or strategy == "SCMI_WITHSOFT":
        strategy_args["scmi_function"] = sf
        strategy_args["optimizer"] = "LazyGreedy"
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        for_private_set = getPrivateSet(train_set, sel_cls_idx)
        strategy_sel = SCMI(
            train_set,
            unlabeled_lake_set,
            for_query_set,
            for_private_set,
            model,
            num_cls,
            strategy_args,
        )

    if strategy == "random":
        strategy_sel = RandomSampling(
            train_set, unlabeled_lake_set, model, num_cls, strategy_args
        )
    if strategy == "WASSAL" or strategy == "WASSAL_WITHSOFT":
        for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
        strategy_sel = WASSAL_Multiclass(
            train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args
        )

    if strategy == "WASSAL_P" or strategy == "WASSAL_P_WITHSOFT":
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        for_private_set = getPrivateSet(train_set, sel_cls_idx)
        strategy_sel = WASSAL_P(
            train_set,
            unlabeled_lake_set,
            for_query_set,
            for_private_set,
            model,
            num_cls,
            strategy_args,
        )

    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
    optimizer = optimizer_without_scheduler(model, learning_rate)
    final_val_predictions = []
    final_val_classifications = []
    final_tst_predictions = []
    final_tst_classifications = []

    for i in range(num_rounds):
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0

        if i == 0:
            print("Initial training epoch")
            if os.path.exists(
                initModelPath
            ):  # Read the initial trained model if it exists
                model.load_state_dict(torch.load(initModelPath, map_location=device))
                print(
                    "Init model loaded from disk, skipping init training: ",
                    initModelPath,
                )
                model.eval()
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(
                            predicted.eq(targets).cpu().numpy()
                        )

                    final_tst_predictions = []
                    final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        final_tst_predictions += list(predicted.cpu().numpy())
                        final_tst_classifications += list(
                            predicted.eq(targets).cpu().numpy()
                        )
                    best_val_acc = val_correct / val_total
                    val_acc[i] = val_correct / val_total
                    tst_acc[i] = tst_correct / tst_total
                    val_losses[i] = val_loss
                    tst_losses[i] = tst_loss
                    res_dict["test_acc"].append(tst_acc[i] * 100)
                continue
            else:  
                print("Training initial model from scratch...")
                
                # Create trainloader for initial training
                trainloader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=trn_batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2
                )
                
                # Train until convergence or max epochs
                num_ep = 0
                while full_trn_acc[i] < 0.99 and num_ep < 50:
                    num_ep += 1
                    model.train()
                    
                    # Training loop
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                    # Compute training accuracy after each epoch
                    model.eval()
                    full_trn_correct = 0
                    full_trn_total = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)
                            full_trn_total += targets.size(0)
                            full_trn_correct += predicted.eq(targets).sum().item()
                    
                    full_trn_acc[i] = full_trn_correct / full_trn_total
                    print(f"Initial training epoch [{num_ep}] Training Acc: {full_trn_acc[i]:.4f}")
                
                # Save the trained initial model
                torch.save(model.state_dict(), initModelPath)
                print(f"Initial model saved to {initModelPath}")
                
                # Now evaluate on validation and test sets
                model.eval()
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(predicted.eq(targets).cpu().numpy())

                    final_tst_predictions = []
                    final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        final_tst_predictions += list(predicted.cpu().numpy())
                        final_tst_classifications += list(predicted.eq(targets).cpu().numpy())
                
                val_acc[i] = val_correct / val_total
                tst_acc[i] = tst_correct / tst_total
                val_losses[i] = val_loss
                tst_losses[i] = tst_loss
                res_dict["test_acc"].append(tst_acc[i] * 100)
                continue
        else:
            # Remove true labels from the unlabeled dataset, the hypothesized labels are computed when select is called
            unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
            print(
                "Started a new AL round, and updating the model, queryset and data(train and unlabeled_set) for strategy "
                + sf
            )
            strategy_sel.update_data(train_set, unlabeled_lake_set)
            strategy_sel.update_model(model)

           
            ####SIM####
            if (
                strategy == "SIM"
                or strategy == "SIM_WITHSOFT"
                or strategy == "WASSAL"
                or strategy == "WASSAL_WITHSOFT"
            ):
                if('WASSAL' in strategy):
                    for_query_set = getQuerySet(train_set, sel_cls_idx,recipe="asis")
                else:
                # make a dataloader for the misclassifications - only for experiments with targets
                    for_query_set = getQuerySet(train_set, sel_cls_idx)
                print("updating queryset for strategy " + sf)
                strategy_sel.update_queries(for_query_set)

                print("size of query set", len(for_query_set))
            # if SCMI_WITHSOFT
            elif (
                strategy == "SCMI_WITHSOFT"
                or strategy == "SCMI"
                or strategy == "WASSAL_P"
                or strategy == "WASSAL_P_WITHSOFT"
            ):
                if sf.endswith("mi"):
                    if feature == "classimb":
                        # make a dataloader for the misclassifications - only for experiments with targets
                        for_query_set = getQuerySet(train_set, sel_cls_idx)
                        for_private_set = getPrivateSet(train_set, sel_cls_idx)
                        print("updating queryset for strategy " + sf)
                        strategy_sel.update_queries(for_query_set)
                        strategy_sel.update_privates(for_private_set)

                        print("size of query set", len(for_query_set))
            # if AL_WITHSOFT
            elif strategy == "AL_WITHSOFT" or strategy == "AL":
                if sf == "glister-tss" or sf == "gradmatch-tss":
                    for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
                    print("updating queryset for strategy " + sf)
                    strategy_sel.update_queries(for_query_set)

                    print("size of query set", len(for_query_set))
            
            # compute the error log before every selection
            if computeErrorLog:
                tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(
                    test_set,
                    val_set,
                    final_val_classifications,
                    final_val_predictions,
                    final_tst_classifications,
                    final_tst_predictions,
                    all_logs_dir,
                    sf + "_" + str(bud),
                )

                csvlog.append([100 - x for x in tst_err_log])
                val_csvlog.append([100 - x for x in val_err_log])

            #update softsubset model and query if WITHSOFT
            if "WITHSOFT" in strategy or strategy=="WASSAL":
                print(
                    "Updating softsoft data, queryset and model for strategy " + sf,
                )

                strategy_softsubset.update_data(train_set, unlabeled_lake_set)
                strategy_softsubset.update_model(model)

                for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
                strategy_softsubset.update_queries(for_query_set)
                
            classwise_final_indices_simplex = None
           
            #get simplex_query
            if "WITHSOFT" in strategy or strategy== "WASSAL":
                print(
                        "Calculating simplexes since we need to do softsubsetting for strategy "
                        + sf
                )
    
                subset,classwise_final_indices_simplex = strategy_softsubset.select(budget)
                #for WASSAL classwise_final_indices_simplex is none so below steps not needed
                if classwise_final_indices_simplex is not None:
                    
                    simplex_dir = os.path.join(all_logs_dir, "simplex")
                    subprocess.run(["mkdir", "-p", simplex_dir])
                    classwise_final_indices_simplex_cpu = []

                    for (
                        selected_indices,
                        simplex_query,
                        class_idx,
                    ) in classwise_final_indices_simplex:
                        # Move tensors to CPU if they're on CUDA
                        if isinstance(simplex_query, torch.Tensor):
                            simplex_query_cpu = simplex_query.cpu()
                        else:
                            simplex_query_cpu = simplex_query
                        
                        classwise_final_indices_simplex_cpu.append((
                            selected_indices,
                            simplex_query_cpu,
                            class_idx,
                        ))

                    # Now call with CPU tensors
                    plotsimpelxDistribution(
                        lake_set, classwise_final_indices_simplex_cpu,simplex_dir
                    )

            weighted_lakeloader = None
            # Eq 7: build the soft-subset weighted loader (mirrors SVHN/CIFAR-10/
            # Pneumonia drivers). Toggleable since Caltech originally never
            # implemented this at all - "WASSAL_WITHSOFT" behaved identically
            # to plain WASSAL for this dataset.
            if PAPER_ALIGNED_SOFT_LOSS and 'WITHSOFT' in strategy and classwise_final_indices_simplex is not None:
                all_small_images = []
                all_small_targets = []
                all_small_simplex_query = []

                # Each class's simplex is normalized fully independently
                # (_proj_simplex has no joint constraint across classes),
                # so nothing otherwise stops the same lake point from
                # being a top-weight "landmark" for more than one of
                # Caltech's 102 classes at once, which would feed the
                # model contradictory pseudo-labels for the same image in
                # the same optimizer step. (S1 exclusion needs no
                # handling here: the strategy class already zeroes S1
                # positions in simplex_query before returning it -
                # verified directly.) Assign each point to at most one
                # class - whichever class it has the highest weight
                # under.
                all_class_weights = torch.stack(
                    [cw[0].detach().cpu() for cw in classwise_final_indices_simplex]
                )
                argmax_class_per_point = all_class_weights.argmax(dim=0)

                for class_pos, (
                    simplex_query,
                    simplex_refrain,
                    class_idx,
                ) in enumerate(classwise_final_indices_simplex):
                    images = [lake_set[i][0] for i in range(len(lake_set))]
                    targets = torch.tensor(class_idx)
                    targets = targets.repeat(len(lake_set))
                    sofftsimplex_query = simplex_query.detach().cpu().numpy()
                    eligible = (argmax_class_per_point == class_pos).numpy()
                    sofftsimplex_query = sofftsimplex_query * eligible
                    # top_elements_contribute_to_percentage's target_sum is a
                    # fixed 0.8, i.e. it assumes the input already sums to
                    # ~1 (true for an unmasked, freshly-projected simplex).
                    # Zeroing ineligible entries breaks that assumption -
                    # renormalize over the eligible mass so the 80% cutoff,
                    # and the budget truncation, only ever pick from truly
                    # eligible (argmax-matching) points.
                    eligible_mass = sofftsimplex_query.sum()
                    if eligible_mass <= 0:
                        continue
                    sofftsimplex_query = sofftsimplex_query / eligible_mass
                    _, top_n_indices = top_elements_contribute_to_percentage(
                        sofftsimplex_query, ss_max_budget_percentage, ss_budget
                    )
                    # Hard post-filter, checking the actual renormalized
                    # weight rather than the class mask: top_elements_
                    # contribute_to_percentage's cumulative-sum loop can,
                    # at the extreme n_percent=100 edge, fail to break
                    # before spilling into the zero-valued tail due to
                    # floating-point rounding after renormalization -
                    # verified by direct testing (1550+ trials against
                    # this exact function). Filtering on the weight
                    # itself (not just class-mask membership) also
                    # covers S1 points correctly even in that edge case,
                    # without needing to track S1 here at all.
                    top_n_indices = [idx for idx in top_n_indices if sofftsimplex_query[idx] > 0]
                    all_small_images += [images[i] for i in top_n_indices]
                    all_small_targets += targets[top_n_indices.copy()].tolist()
                    all_small_simplex_query += sofftsimplex_query[
                        top_n_indices
                    ].tolist()

                print("size of simplex_query for strategy "+sf+" and budget "+str(budget)+" is "+str(len(all_small_simplex_query))+" in round "+str(i))

                if len(all_small_images) > 0:
                    all_small_targets = torch.tensor(all_small_targets)
                    all_small_simplex_query = torch.tensor(all_small_simplex_query)
                    weighted_lake_set = WeightedDataset(
                        all_small_images,
                        all_small_targets,
                        all_small_simplex_query,
                        None,
                        None,
                    )
                    weighted_lakeloader = torch.utils.data.DataLoader(
                        weighted_lake_set,
                        batch_size=trn_batch_size,
                        shuffle=True,
                        pin_memory=True,
                    )

            #selecting subset using an AL strategy
            # subset = []
            # print("Selecing AL data for strategy " + sf)
            # if strategy == "WASSAL" or strategy == "WASSAL_WITHSOFT":
            #     print('selecting subset as well for '+sf)
                
            #     for (
            #         selected_indices,
            #         simplex_query,
            #         simplex_refrain,
            #         class_idx,
            #     ) in classwise_final_indices_simplex:
            #         subset += selected_indices

            #     # analyze_simplex(temp_args,lake_set,simplex_query)
            if strategy == "WASSAL_P" or strategy == "WASSAL_P_WITHSOFT":
                subset, simplex_query, simplex_private = strategy_sel.select(budget)

            # for other strategies simple to get subset
            elif "WASSAL" not in strategy:
                subset = strategy_sel.select(budget)

            lake_subset_idxs = (
                subset  # indices wrt to lake that need to be removed from the lake
            )

            perClsSel = getPerClassSel(true_lake_set, lake_subset_idxs, num_cls)
            res_dict["sel_per_cls"].append(perClsSel)

            train_set, lake_set, true_lake_set, add_val_set = aug_train_subset(
                train_set,
                lake_set,
                true_lake_set,
                subset,
                lake_subset_idxs,
                budget,
                True,
            )  # aug train with random if budget is not filled
            print(
                "After augmentation, size of train_set: ",
                len(train_set),
                " unlabeled set: ",
                len(lake_set),
                " val set: ",
                len(val_set),
            )

            # Create a new trainloader with augmented training set
            trainloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=trn_batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )

            # Training
            start_time = time.time()
            num_ep = 0
            while full_trn_acc[i] < 0.99 and num_ep < 50:
                num_ep += 1
                model.train()
                if PAPER_ALIGNED_SOFT_LOSS and weighted_lakeloader is not None:
                    # Eq 7+8: one optimizer step per epoch, combining hard
                    # loss over train_set with the soft-weighted loss over
                    # the soft-subset (per-sample weighted, raw sum).
                    # Caltech's 102 classes make the soft-subset far larger
                    # than SVHN/CIFAR-10/Pneumonia's, so gradients are
                    # accumulated per-batch (backward per batch, step once)
                    # instead of retaining every batch's graph at once -
                    # mathematically identical, bounded peak memory.
                    optimizer.zero_grad()
                    soft_loss_total = 0.0
                    for batch_idx, (inputs, targets, simplex_query, _, _) in enumerate(weighted_lakeloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        simplex_query = simplex_query.to(device)
                        soft_outputs = model(inputs)
                        target_loss_per_sample = criterion_nored(soft_outputs, targets)
                        batch_soft_loss = (simplex_query * target_loss_per_sample).sum()
                        (soft_loss_hyperparam * batch_soft_loss).backward()
                        soft_loss_total += batch_soft_loss.item()
                    hard_loss_total = 0.0
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        batch_hard_loss = criterion(outputs, targets)
                        batch_hard_loss.backward()
                        hard_loss_total += batch_hard_loss.item()
                    optimizer.step()
                else:
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                # Compute training accuracy after each epoch
                model.eval()
                full_trn_correct = 0
                full_trn_total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        full_trn_total += targets.size(0)
                        full_trn_correct += predicted.eq(targets).sum().item()
                
                full_trn_acc[i] = full_trn_correct / full_trn_total
                print(f"Selection Epoch {i}  Training epoch [ {num_ep} ]  Training Acc:  {full_trn_acc[i]}")

            timing[i] = time.time() - start_time
            print("Round ", i, " training time: ", timing[i])

            # Compute the training loss and accuracy
            model.eval()
            full_trn_loss = 0
            full_trn_correct = 0
            full_trn_total = 0

            # Create a full training loader for evaluation
            full_trainloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=trn_batch_size,
                shuffle=False,
                pin_memory=True,
            )

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(full_trainloader):
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    full_trn_total += targets.size(0)
                    full_trn_correct += predicted.eq(targets).sum().item()

                full_trn_acc[i] = full_trn_correct / full_trn_total

                # Compute validation and test accuracy
                final_val_predictions = []
                final_val_classifications = []
                for batch_idx, (inputs, targets) in enumerate(valloader):
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    final_val_predictions += list(predicted.cpu().numpy())
                    final_val_classifications += list(
                        predicted.eq(targets).cpu().numpy()
                    )

                final_tst_predictions = []
                final_tst_classifications = []
                for batch_idx, (inputs, targets) in enumerate(
                    tstloader
                ):  # Compute test accuracy
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    tst_loss += loss.item()
                    _, predicted = outputs.max(1)
                    tst_total += targets.size(0)
                    tst_correct += predicted.eq(targets).sum().item()
                    final_tst_predictions += list(predicted.cpu().numpy())
                    final_tst_classifications += list(
                        predicted.eq(targets).cpu().numpy()
                    )
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            full_val_acc = list(np.array(val_acc))
            full_timing = list(np.array(timing))
            res_dict["test_acc"].append(tst_acc[i] * 100)
            print(
                "Epoch:",
                i + 1,
                "FullTrn,TrainAcc,ValLoss,ValAcc,TstLoss,TstAcc,Time:",
                full_trn_loss,
                full_trn_acc[i],
                val_loss,
                val_acc[i],
                tst_loss,
                tst_acc[i],
                timing[i],
            )
            print(
                "Gain in accuracy: ",
                res_dict["test_acc"][i] - res_dict["test_acc"][i - 1],
            )

        
        # Save initial model after first epoch
        if i == 0:
            print("Saving initial model")
            torch.save(
                model.state_dict(), initModelPath
            )  # save initial train model if not present

    # Compute the statistics of the final model
    if computeErrorLog:
        print("**** Final Metrics after Targeted Learning ****")
        tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(
            test_set,
            val_set,
            final_val_classifications,
            final_val_predictions,
            final_tst_classifications,
            final_tst_predictions,
            all_logs_dir,
            sf + "_" + str(bud),
        )
        csvlog.append([100 - x for x in tst_err_log])
        val_csvlog.append([100 - x for x in val_err_log])
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog
        with open(os.path.join(all_logs_dir, exp_name + ".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(csvlog)
    
    # save results dir with test acc and per class selections
    with open(os.path.join(all_logs_dir, exp_name + ".json"), "w") as fp:
        json.dump(res_dict, fp)
    
    # Print overall acc improvement and rare class acc improvement
    print_final_results(res_dict, sel_cls_idx)
    print("Total gain in accuracy: ", res_dict["test_acc"][i] - res_dict["test_acc"][0])
    # ============= MISSING SECTION ENDS HERE =============
    
    
#     tsne_plt.show()


# %%
# exp1 (seed 48) was the initial single-seed sweep; exp2-4 follow the same
# seed convention as the other datasets.
experiments = ["exp2", "exp3", "exp4"]
seeds = [48, 86, 28, 92]
budgets = [25,50,100,175,200]

embedding_type = "features"  # Type of the representation to use (gradients/features)
model_name = "ResNet18"  # Model to use for training
initModelPath = (
    str(RESULTS_DIR) + "/onlywassal"
    # + experiment_name
    # + "/"
    + data_name
    + "_"
    + model_name
    + "_"
    + embedding_type
    + "_"
    + str(learning_rate)
   
)
#skip strategies that are already run
skip_strategies = []
skip_budgets = []
only_methods = []
soft_loss_hyperparam=3

if __name__ == "__main__":
    # Accept skip_strategies and skip_budgets from command line arguments
    skip_strategies = sys.argv[1].split()
    skip_methods= sys.argv[2].split()
    skip_budgets = list(map(int, sys.argv[3].split()))
    soft_loss_hyperparam=float(sys.argv[6])
    if len(sys.argv) > 10 and sys.argv[10].strip():
        # optional whitelist: run ONLY these methods (comma-separated)
        only_methods = sys.argv[10].split()

# Model Creation
model = create_model(model_name, num_cls, device, embedding_type)
strategies = [
    # al soft
    ("WASSAL", "WASSAL"),
    ("WASSAL_WITHSOFT", "WASSAL_WITHSOFT"),
    ("AL", "glister"),
    ("AL_WITHSOFT", "glister_withsoft"),
    ("AL", "gradmatch-tss"),
    ("AL_WITHSOFT", "gradmatch-tss_withsoft"),
    ("AL", "coreset"),
    ("AL_WITHSOFT", "coreset_withsoft"),
    ("AL", "leastconf"),
    ("AL_WITHSOFT", "leastconf_withsoft"),
    ("AL", "margin"),
    ("AL_WITHSOFT", "margin_withsoft"),
    ("random", "random"),
    ("AL", "badge"),
    ("AL", "badge_withsoft"),
    ("AL_WITHSOFT", "us_withsoft"),
    ("AL", "us"),
    # modern (2022-2024) baselines for up-to-date comparison
    ("AL", "typiclust"),
    ("AL", "probcover"),
    ("AL", "dcom"),
    ("AL", "alfamargin"),
    ("AL", "maxherding"),
    ("AL", "uherding"),
    ("AL", "wassersteinip"),
   
]

#torch._C._cuda_attach_out_of_memory_observer(torch.cuda.memory._dump_snapshot("my_snapshot.pickle"))
for i, experiment in enumerate(experiments):
    seed = seeds[i]
    torch.manual_seed(seed)
    np.random.seed(seed)
    run = experiment

    # Loop for each budget from 50 to 400 in intervals of 50
    for b in budgets:
        # Loop through each strategy
        for strategy, method in strategies:
            #skip strategies that are already run
            if strategy in skip_strategies:
                continue
            if method in skip_methods and b in skip_budgets:
                continue
            if only_methods and method not in only_methods:
                continue
            print("Budget ", b, " Strategy ", strategy, " Method ", method)
            run_targeted_selection(
                data_name,
                datadir,
                feature,
                model_name,
                b,  # updated budget
                split_cfg,
                learning_rate,
                run,
                device,
                computeClassErrorLog,
                strategy,
                method,
                embedding_type,
                soft_loss_hyperparam

            )