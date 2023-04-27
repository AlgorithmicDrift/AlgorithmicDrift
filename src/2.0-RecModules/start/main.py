import sys

import torch.cuda
from run_processes import run_processes
import numpy as np
import torch
import os
import time

path = "../../new_normal_data_500/processed/"
folder = "SyntheticDataset/History_80_10/"

gpu_id = "0"

user_count_start_args = "0"
user_count_end_args = "500"

module = "evaluation"  # training, evaluation, generation, merge_rec_sessions, recbole_dataset
dataset = None

if len(sys.argv) >= 3:
    dataset = sys.argv[1]
    module = sys.argv[2]

# "No_strategy", "No_feedbackloop_no_strategy", "Preprocessing", "Inprocessing_adversarial", "Inprocessing_penalization",
# "Postprocessing", "Organic", "Sparsity"
strategy = "No_strategy"

if strategy == "Organic":
    module = "generation"

to_parallelize = "models"  # "eta", "factors", "sub_strategies", "topk", "models"

# models RecVAE, MultiVAE, LightGCN, NGCF, UserKNN
factor_name = "Lambda"  # "Rewiring", "Lambda", "Factor"

if strategy == "Preprocessing":
    factor_name = "Rewiring"

elif strategy == "Inprocessing_adversarial":
    factor_name = "Lambda"

elif strategy == "Inprocessing_penalization" or strategy == "Postprocessing":
    factor_name = "Lambda"

elif strategy == "Sparsity":
    factor_name = "Factor"

# choose the etas, one factor (and one sub strategy)
if to_parallelize == "eta":
    eta_args = ["0.65_0.2_0.15", "0.7_0.2_0.1", "0.75_0.2_0.05", "0.79_0.2_0.01",
                "0.45_0.4_0.15", "0.5_0.4_0.1", "0.55_0.4_0.05", "0.59_0.4_0.01",
                "0.25_0.6_0.15", "0.3_0.6_0.1", "0.35_0.6_0.05", "0.39_0.6_0.01"]
    factors_args = np.repeat("{}_0.001".format(factor_name), len(eta_args))
    sub_strategies_args = np.repeat("Harmful_dynamic", len(eta_args))
    topk_args = np.repeat("10", len(eta_args))
    models_args = np.repeat("UserKNN", len(eta_args))

# choose the factors, one eta (and one sub strategy)
elif to_parallelize == "factors":
    # factors_args = [
    #     "Lambda_0.2",
    #     "Lambda_0.3",
    #     "Lambda_0.4",
    #     "Lambda_0.5",
    #     "Lambda_0.6",
    #     "Lambda_0.7",
    #     "Lambda_0.8",
    #     "Lambda_0.9"]
    # factors_args = [
    #     "Lambda_0.5",
    #     "Lambda_0.6",
    #     "Lambda_0.7",
    #     "Lambda_0.8",
    #     "Lambda_0.9"]
    factors_args = [
        "Lambda_5",
        "Lambda_10",
        "Lambda_15",
        "Lambda_20",
        "Lambda_25",
    ]
    # factors_args = [
    #     "Lambda_0.000005",
    #     "Lambda_0.000006",
    #     "Lambda_0.000007",
    #     "Lambda_0.000008",
    #     "Lambda_0.000009",
    #     "Lambda_0.00001"]
    # factors_args = [
    #     "Lambda_0.001",
    #     "Lambda_0.0025",
    #     "Lambda_0.005",
    #     "Lambda_0.0075",
    #     "Lambda_0.01"]
    # factors_args = ["Lambda_25", "Lambda_50", "Lambda_75", "Lambda_125", "Lambda_150"]
    # factors_args = ["Rewiring_10", "Rewiring_25", "Rewiring_50", "Rewiring_75", "Rewiring_90"]

    eta_args = np.repeat("0.3", len(factors_args))
    sub_strategies_args = np.repeat("Random", len(factors_args))
    topk_args = np.repeat("10", len(factors_args))
    models_args = np.repeat("NGCF", len(factors_args))

# Preprocessing sub strategies
# Random, Similarity, Popularity, Naive_Random, Naive_Similarity, Naive_Popularity
elif to_parallelize == "sub_strategies":  # choose the sub strategies, one eta and one factor
    sub_strategies_args = [
        "Random",
        "Random",
        "Random",
        "Random",
        "Random",
        "Random"]
    factors_args = np.repeat(
        "{}_75".format(factor_name),
        len(sub_strategies_args))
    eta_args = np.repeat("0.5", len(sub_strategies_args))
    topk_args = np.repeat("10", len(sub_strategies_args))
    models_args = np.repeat("RecVAE", len(sub_strategies_args))

elif to_parallelize == "topk":
    topk_args = ["1", "5", "10", "25", "50"]
    factors_args = np.repeat("{}_75".format(factor_name), len(topk_args))
    sub_strategies_args = np.repeat("Harmful_static", len(topk_args))
    eta_args = np.repeat("0.5", len(topk_args))
    models_args = np.repeat("Pop", len(topk_args))

elif to_parallelize == "models":
    models_args = ["RecVAE", "LightGCN", "MultiVAE", "NGCF", 'UserKNN']
    factors_args = np.repeat("{}_75".format(factor_name), len(models_args))
    sub_strategies_args = np.repeat("Random", len(models_args))
    eta_args = np.repeat("0.79_0.2_0.01", len(models_args)) if dataset is None \
                                        else np.repeat(dataset, len(models_args))
    topk_args = np.repeat("10", len(models_args))

retrain_args = np.repeat("0", len(eta_args))

# "to_parallelize" to call in parallel (each row will be called in parallel)
indices_call = [[0, 1, 2, 3, 4]]#, [4, 5, 6, 7], [8, 9, 10, 11]]  # 2, 3, 4]]#, 5]]#, 6, 7, 8]]

if len(eta_args) != len(sub_strategies_args):
    print("Check arrays")
    exit(1)

run_processes(
    path,
    folder,
    models_args,
    module,
    eta_args,
    strategy,
    factors_args,
    sub_strategies_args,
    topk_args,
    retrain_args,
    user_count_start_args,
    user_count_end_args,
    gpu_id,
    indices_call)
