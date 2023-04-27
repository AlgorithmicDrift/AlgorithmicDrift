import sys

import torch.cuda
from run_processes import run_processes
import numpy as np
import torch
import os
import time

path = "../../new_data/processed/"
folder = "SyntheticDataset/History_80_10/"

gpu_id = "0"

user_count_start_args = "0"
user_count_end_args = "500"

module = "training"  # training, evaluation, generation, merge_rec_sessions, recbole_dataset
dataset = None

if len(sys.argv) >= 3:
    dataset = sys.argv[1]
    module = sys.argv[2]

# "No_strategy", "Organic"
strategy = "No_strategy"

if strategy == "Organic":
    module = "generation"

to_parallelize = "models"  # "eta", "models"

# choose the etas, one factor (and one sub strategy)
if to_parallelize == "eta":
    eta_args = ["0.65_0.2_0.15", "0.7_0.2_0.1", "0.75_0.2_0.05", "0.79_0.2_0.01",
                "0.45_0.4_0.15", "0.5_0.4_0.1", "0.55_0.4_0.05", "0.59_0.4_0.01",
                "0.25_0.6_0.15", "0.3_0.6_0.1", "0.35_0.6_0.05", "0.39_0.6_0.01"]
    models_args = np.repeat("UserKNN", len(eta_args))

elif to_parallelize == "models":
    models_args = ["RecVAE"]#, "LightGCN", "MultiVAE", "NGCF", 'UserKNN']
    eta_args = np.repeat("0.79_0.2_0.01", len(models_args)) if dataset is None \
                                        else np.repeat(dataset, len(models_args))


# "to_parallelize" to call in parallel (each row will be called in parallel)
indices_call = [[0]]#, 1, 2, 3, 4]]#, [4, 5, 6, 7], [8, 9, 10, 11]]  # 2, 3, 4]]#, 5]]#, 6, 7, 8]]


run_processes(
    path,
    folder,
    models_args,
    module,
    eta_args,
    strategy,
    user_count_start_args,
    user_count_end_args,
    gpu_id,
    indices_call)
