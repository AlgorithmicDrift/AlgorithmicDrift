import sys
import os

import numpy as np
import torch.cuda

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from os.path import exists
import logging
import pandas as pd
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from recbole.data.utils import *
from graph_generation import generate_graphs, generate_organic_graphs, create_T_tensor
from utils.load_data import get_dataset, create_dataset_recbole
from utils.model_utils import load_model, train_model, get_parameter_dict, rewire_train
from utils.data_utils import create_folders, get_dataset_name_and_paths, get_parsed_args

args = get_parsed_args(sys.argv)
print(args)

torch.cuda.set_device(int(args["gpu_id"]))

dataset_path, dataset_name, saving_path = get_dataset_name_and_paths(args)

print(dataset_path + dataset_name, saving_path)

history_dataset, uir_dataset, utils_dicts = get_dataset(
    dataset_path + dataset_name + ".tsv")

videos_labels_dict, videos_slants_dict, reverse_users_dict, reverse_videos_dict = utils_dicts

users = list(set([i[0] for i in uir_dataset]))

df = pd.DataFrame(
    uir_dataset,
    columns=[
        "User",
        "Item",
        "Rating",
        "Orientation",
        "Label"])
df["Label"] = np.where(df["Label"] == "harmful", 1, 0)

df = df.sort_values(by=["Item"])
items = df["Item"].unique()
items_labels = df[["Item", "Label"]].groupby(
    "Item").mean().to_numpy().reshape(-1)

items_recbole = [0] + list(np.array(items) + 1)
items_counts_non_rad = np.array(
    list(zip(items_recbole, np.zeros(len(items_recbole), dtype=int))))

df = df.sort_values("User")
mean_slant_users = df[["User", "Label"]].groupby("User").mean().to_numpy()
non_rad_users = df[df["Orientation"] == 'non radicalized']["User"].unique()
semi_rad_users = df[df["Orientation"] == 'semi-radicalized']["User"].unique()

# add fake user and fake item
mean_slant_users = np.insert(mean_slant_users, 0, 0.0)
items_labels = np.insert(items_labels, 0, 0)

num_interactions = len(uir_dataset)
num_users = len(users)
num_items = len(items)

# remove "orientation" and "label" columns
uir_dataset = list(np.array(uir_dataset)[:, :3])

print("Dataset interactions:", num_interactions)
print("Dataset users:", num_users)
print("Dataset items:", num_items)

if args["module"] == "recbole_dataset":
    create_dataset_recbole(args, uir_dataset)

    if args["strategy"] != "Organic":
        create_folders(saving_path, args)
        print("Recbole dataset created")
    exit(0)

if args["module"] == "training":
    create_folders(saving_path, args)

model_checkpoint_folder = saving_path + args["model"] + "/model_checkpoint/"
if args["strategy"] == 'Postprocessing' or args["strategy"] == 'No_feedbackloop_no_strategy':
    model_checkpoint_folder = dataset_path + \
                              "No_strategy/" + args["model"] + "/model_checkpoint/"

print("MODEL CHECKPOINT:", model_checkpoint_folder)

parameter_dict = get_parameter_dict(args, model_checkpoint_folder)

dataset = "eta_{}".format(args["eta"]) if args["folder"].startswith("SyntheticDataset") else "youtube"

if args["strategy"].startswith("Inprocessing") or args["strategy"] == "Sparsity":
    dataset = dataset + \
              "_{}".format(args["factor"])

elif args["strategy"] == "Preprocessing" or args["strategy"] == "Postprocessing":
    dataset = dataset + \
              "_{}_{}".format(args["factor"], args["sub_strategy"])

if args["model"] == "UserKNN":
    model = "ItemKNN"  # because in recbole the name is ItemKNN
elif args["model"] == "Random":
    model = "RecVAE"  # or whatever
else:
    model = args["model"]

config = Config(
    model=model,
    dataset=dataset,
    config_dict=parameter_dict)

# SET CONFIG DEVICE
config["device"] = "cuda"


# init random seed
init_seed(config["seed"], config["reproducibility"])

# logger initialization
init_logger(config)
logger = getLogger()

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

density = num_interactions / (num_users * num_items)
print("Dataset density:", np.around(density, decimals=3))

transient_nodes = [reverse_videos_dict[x]
                   for x in items if videos_labels_dict[x] == "harmful"]
neutral_nodes = [reverse_videos_dict[x]
                 for x in items if videos_labels_dict[x] == "neutral"]

print("Percentage harmful on", num_items, "videos:", np.around(
    len(transient_nodes) / num_items, decimals=2) * 100, "%")

if args["module"] == "training":

    train_model(
        args=args,
        model_checkpoint_folder=model_checkpoint_folder,
        config=config,
        users=users,
        logger=logger,
        num_users=num_users,
        num_items=num_items,
        utils_dicts=utils_dicts,
        non_rad_users=non_rad_users,
        semi_rad_users=semi_rad_users,
        mean_slant_users=mean_slant_users,
        items_labels=items_labels,
        saving_path=saving_path)

elif args["module"] == "evaluation":

    print("\n Evaluation on test:")

    model, data, _, checkpoint_path, checkpoint_file, trainer = load_model(
        model_checkpoint_folder=model_checkpoint_folder,
        config=config, logger=logger, utils_dicts=utils_dicts,
        non_rad_users=non_rad_users, semi_rad_users=semi_rad_users, users=users,
        num_users=num_users, num_items=num_items, transient_nodes=transient_nodes,
        history_dataset=history_dataset, items_labels=items_labels,
        args=args, saving_path=saving_path)

    _, _, test_data = data

    result = trainer.evaluate(
        test_data,
        model_file=checkpoint_path,
        show_progress=False)

    print(result)

    evaluation_filename = "evaluation.tsv"
    evaluation_path = saving_path + args["model"] + "/"

    print(evaluation_path + evaluation_filename)

    df = pd.DataFrame([result], columns=result.keys())
    df.to_csv(evaluation_path + evaluation_filename, sep="\t", index=False)

elif args["module"] == "generation":
    logger.setLevel(logging.ERROR)  # just to avoid too many prints

    topk = args["topk"]

    graphs_folder = saving_path + args["model"] + "/graphs/topk_" + str(topk) + "/"

    print(graphs_folder)

    if args["retrain"]:
        graphs_folder = graphs_folder[:-1] + "_retrain/"

    B = 50  # 100
    d = 100  # 200

    if args["strategy"] == "Preprocessing":  # rewiring is necessary
        _, _, history_dataset, _, _, _ = load_model(
            model_checkpoint_folder=model_checkpoint_folder,
            config=config, logger=logger, utils_dicts=utils_dicts,
            non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
            args=args, saving_path=saving_path)

    T_tensor = create_T_tensor(
        history_dataset,
        user_count_start=args["user_count_start"],
        user_count_end=args["user_count_end"],
        num_items=num_items)

    if args["strategy"] == "Organic":
        graphs_folder = saving_path + "graphs/"
        if not exists(graphs_folder):
            os.makedirs(graphs_folder)
        generate_organic_graphs(T_tensor, history_dataset, dataset_name, dataset_path, B, d, items_labels,
                                reverse_users_dict, reverse_videos_dict, transient_nodes,
                                saving_path, graphs_folder, args)
    else:
        generate_graphs(
            T_tensor,
            history_dataset,
            dataset_name,
            B,
            d,
            topk=topk,
            num_items=num_items,
            num_users=num_users,
            items_labels=items_labels,
            users=users,
            non_rad_users=non_rad_users,
            semi_rad_users=semi_rad_users,
            saving_path=saving_path,
            reverse_users_dict=reverse_users_dict,
            reverse_videos_dict=reverse_videos_dict,
            model_checkpoint_folder=model_checkpoint_folder,
            config=config,
            logger=logger,
            transient_nodes=transient_nodes,
            graphs_folder=graphs_folder,
            args=args,
            utils_dicts=utils_dicts)
