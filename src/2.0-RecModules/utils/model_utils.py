import sys
import os

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import logging
import copy
import torch
from recbole.trainer.trainer import Trainer, RecVAETrainer
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
import pandas as pd
from os.path import exists

from models.custom_trainers.item_embeddings_adversarial_trainer import ItemEmbeddingsAdversarialTrainer
from models.custom_trainers.mitigation_postprocessing_trainer import MitigationPostprocessingTrainer
from models.userknn import UserKNN
from models.lightgcn import LightGCN
from models.recvae import RecVAE
from models.multivae import MultiVAE
from models.pop import Pop
from models.ngcf import NGCF
from models.random import Random
from utils.rewiring_strategy import start_rewiring

from recbole.data.utils import *


def get_parameter_dict(args, model_checkpoint_folder):
    model = args["model"]
    path = args["path"]
    folder = args["folder"]

    parameter_dict = {
        "eval_step": 25,
        "topk": [10, 20],
        "metrics": ["Recall", "nDCG", "Hit"],
        "valid_metric": "Recall@10",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
        "data_path": path + folder + "recbole",
        "checkpoint_dir": model_checkpoint_folder,
        "epochs": 100,
    }

    if model == "RecVAE":
        parameter_dict["neg_sampling"] = None
        parameter_dict["hidden_dimension"] = 512
        parameter_dict["latent_dimension"] = 512
        parameter_dict["epochs"] = 100

    elif model == "MultiVAE":
        parameter_dict["neg_sampling"] = None
        parameter_dict["hidden_dimension"] = 512
        parameter_dict["latent_dimension"] = 1024
        parameter_dict["epochs"] = 100

    elif model == "LightGCN" or model == "NGCF":
        parameter_dict["neg_sampling"] = {"uniform": 1}
        parameter_dict["embedding_size"] = 1024
        parameter_dict["epochs"] = 200

    elif model == "UserKNN":
        parameter_dict["k"] = 25  # 10
        parameter_dict["method"] = "user"
        parameter_dict["epochs"] = 1

    return parameter_dict


def convert_dataset_to_dataframe(dataset, utils_dicts, non_rad_users, semi_rad_users):
    videos_labels_dict, videos_slants_dict, _, reverse_videos_dict = utils_dicts

    new_dataset = dataset.inter_matrix()
    df = pd.DataFrame({'User': new_dataset.row, 'Item': new_dataset.col})[['User', 'Item']]
    df['Item'] = df['Item'] - 1
    df['User'] = df['User'] - 1

    df = df.astype({'User': int, 'Item': int})

    df = df.assign(Label=np.zeros(len(df['Item'])))
    df = df.assign(Orientation=np.zeros(len(df['Item'])))
    df = df.assign(Slant=np.zeros(len(df['Item'])))

    df.reset_index()
    df = df.sort_values('User')

    orientations = []
    labels = []
    slants = []

    for index, row in df.iterrows():
        item = int(row['Item'])
        labels.append(videos_labels_dict[item])
        slants.append(videos_slants_dict[reverse_videos_dict[item]])

        user = int(row['User'])
        orientation = 'radicalized'
        if user in non_rad_users:
            orientation = 'non radicalized'
        elif user in semi_rad_users:
            orientation = 'semi-radicalized'
        orientations.append(orientation)

    df['Orientation'] = orientations
    df['Slant'] = slants
    df['Label'] = labels

    return df


def rewire_train(data, utils_dicts, non_rad_users, semi_rad_users, args, saving_path=None):
    rewiring_strategy = args["sub_strategy"]
    factor = args["factor"]

    discard_percentage = float("0." + factor.split("_")[1])

    train_data, valid_data, test_data = data

    train_df = convert_dataset_to_dataframe(train_data.dataset, utils_dicts,
                                            non_rad_users, semi_rad_users)

    valid_df = convert_dataset_to_dataframe(valid_data.dataset, utils_dicts,
                                            non_rad_users, semi_rad_users)

    test_df = convert_dataset_to_dataframe(test_data.dataset, utils_dicts,
                                           non_rad_users, semi_rad_users)

    df_for_videos_to_exclude = pd.concat((valid_df, test_df)).reset_index(drop=True)
    rewired_train_df = start_rewiring(train_df, df_for_videos_to_exclude, rewiring_strategy, discard_percentage)

    rewired_history_df = pd.concat((copy.deepcopy(rewired_train_df), valid_df, test_df)).reset_index(drop=True)

    if saving_path is not None:
        videos_labels_dict, videos_slants_dict, reverse_users_dict, reverse_videos_dict = utils_dicts

        rewired_fn = "Histories_eta_{}.tsv".format(args["eta"])
        rewired_path = saving_path + rewired_fn

        reversed_rewired_history_df = rewired_history_df.copy()

        reversed_rewired_history_df["User"] = [reverse_users_dict[x] for x in reversed_rewired_history_df["User"]]
        reversed_rewired_history_df["Item"] = [reverse_videos_dict[x] for x in reversed_rewired_history_df["Item"]]

        # if not exists(rewired_path):
        reversed_rewired_history_df.to_csv(rewired_path, header=["User", "Video", "Label", "Orientation", "Slant"],
                                           sep="\t", index=False)

    history_dataset = {}

    users_group = rewired_history_df.groupby("User")
    for user, _ in users_group:
        interactions = list(users_group.get_group(user)["Item"])
        history_dataset[user] = interactions

    rewired_train_df['User'] = rewired_train_df['User'] + 1
    rewired_train_df['Item'] = rewired_train_df['Item'] + 1

    rewired_train_df = rewired_train_df.assign(item_value=np.ones(len(rewired_train_df)))
    rewired_train_df = rewired_train_df.assign(user_id=rewired_train_df['User'])
    rewired_train_df = rewired_train_df.assign(item_id=rewired_train_df['Item'])

    rewired_train_df = rewired_train_df[['user_id', 'item_id', 'item_value']]

    interaction = Interaction(rewired_train_df)

    train_data.dataset = train_data.dataset.copy(interaction)
    return train_data, history_dataset


def get_model_structure_and_trainer(
        config,
        logger,
        args,
        utils_dicts=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        transient_nodes=None,
        reverse_videos_dict=None,
        history_dataset=None,
        saving_path=None):
    dataset = create_dataset(config)
    # logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_name = args["model"]
    strategy = args["strategy"]
    sub_strategy = args["sub_strategy"]
    module = args["module"]

    factor = args["factor"]

    if strategy == 'Preprocessing':
        data = (train_data, valid_data, test_data)
        train_data, history_dataset = rewire_train(data, utils_dicts, non_rad_users, semi_rad_users, args,
                                                   saving_path=saving_path)

    reweighting_mitigation = True if strategy == 'Inprocessing_penalization' and module == 'training' else False
    postprocessing_mitigation = sub_strategy if strategy == 'Postprocessing' and module == 'evaluation' else ''

    if postprocessing_mitigation != '':
        train_df = convert_dataset_to_dataframe(train_data.dataset, utils_dicts,
                                                non_rad_users, semi_rad_users)
        train_df["Label"] = np.where(train_df["Label"] == "harmful", 1, 0)
        temp_history_dataset = {}

        users_group = train_df.groupby("User")
        for user, _ in users_group:
            interactions = list(users_group.get_group(user)["Item"])
            temp_history_dataset[user] = interactions

    # VAE-based
    if model_name == "MultiVAE" or model_name == "RecVAE":
        if model_name == "MultiVAE":
            model = MultiVAE(
                config=config,
                dataset=train_data.dataset,
                mean_slant_users=mean_slant_users,
                items_labels=items_labels,
                lambda_factor=factor,
                num_users=num_users,
                num_items=num_items,
                reweighting_mitigation=reweighting_mitigation).to(
                config["device"])

            trainer = Trainer(config, model)

        elif model_name == "RecVAE":

            model = RecVAE(
                config=config,
                dataset=train_data.dataset,
                mean_slant_users=mean_slant_users,
                items_labels=items_labels,
                lambda_factor=factor,
                num_users=num_users,
                num_items=num_items,
                reweighting_mitigation=reweighting_mitigation).to(
                config["device"])

            trainer = RecVAETrainer(config, model)

        if postprocessing_mitigation != '':
            trainer = MitigationPostprocessingTrainer(config=config, model=model,
                                                      postprocessing_mitigation=postprocessing_mitigation,
                                                      factor=factor, items_labels=items_labels, num_users=num_users,
                                                      num_items=num_items, users=users, transient_nodes=transient_nodes,
                                                      history_dataset=temp_history_dataset,
                                                      reverse_videos_dict=reverse_videos_dict)

    # GCN-based
    elif model_name == "LightGCN" or model_name == "NGCF":

        if model_name == "LightGCN":
            model = LightGCN(config,
                             train_data.dataset,
                             mean_slant_users=mean_slant_users,
                             items_labels=items_labels,
                             lambda_factor=factor,
                             num_users=num_users,
                             num_items=num_items,
                             reweighting_mitigation=reweighting_mitigation).to(config["device"])
        elif model_name == "NGCF":
            model = NGCF(config, train_data.dataset).to(config["device"])

        if strategy == "Inprocessing_adversarial":

            lambda_factor = int(factor.split("_")[1])
            y = np.array(items_labels, dtype=np.float16)
            y = y.reshape(-1, 1)

            trainer = ItemEmbeddingsAdversarialTrainer(
                config,
                model,
                torch.cuda.FloatTensor(y),
                lambda_factor=lambda_factor)

        elif postprocessing_mitigation != '':
            trainer = MitigationPostprocessingTrainer(config=config, model=model,
                                                      postprocessing_mitigation=postprocessing_mitigation,
                                                      factor=factor, items_labels=items_labels, num_users=num_users,
                                                      num_items=num_items, users=users, transient_nodes=transient_nodes,
                                                      history_dataset=temp_history_dataset,
                                                      reverse_videos_dict=reverse_videos_dict)
        else:
            trainer = Trainer(config, model)

    elif model_name == "Pop":
        model = Pop(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)

    elif model_name == "Random":
        model = Random(num_users, num_items)
        trainer = None

    elif model_name == "UserKNN":
        model = UserKNN(config, train_data.dataset, users).to(config["device"])
        trainer = Trainer(config, model)

    return model, trainer, (train_data, valid_data, test_data), history_dataset


def load_model(
        model_checkpoint_folder,
        config,
        logger,
        args,
        utils_dicts=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        transient_nodes=None,
        reverse_videos_dict=None,
        history_dataset=None,
        saving_path=None):
    model, trainer, data, history_dataset = get_model_structure_and_trainer(
        config=config, logger=logger, args=args, utils_dicts=utils_dicts, users=users,
        mean_slant_users=mean_slant_users, items_labels=items_labels, history_dataset=history_dataset,
        num_users=num_users, num_items=num_items, non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
        transient_nodes=transient_nodes, reverse_videos_dict=reverse_videos_dict, saving_path=saving_path)

    if isinstance(model, Random):
        return model, data, history_dataset, None, None, None

    model_files = os.listdir(model_checkpoint_folder)
    checkpoint_file = model_files[-1]

    checkpoint_path = model_checkpoint_folder + checkpoint_file

    print(checkpoint_path)

    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    # Here you can replace it by your model path.
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])

    return model, data, history_dataset, checkpoint_path, checkpoint_file, trainer


def train_model(
        args,
        model_checkpoint_folder=None,
        interaction_dict=None,
        config=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        logger=None,
        utils_dicts=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        saving_path=None):
    config_temp = copy.deepcopy(config)

    retrain = args["retrain"]
    path = args["path"]
    folder = args["folder"]
    model_name = args["model"]
    eta = args["eta"]

    save_model = True

    if retrain:

        data = []
        for i in range(len(interaction_dict["user_id"])):
            for j in range(len(interaction_dict["item_id"][i])):
                data.append(
                    (interaction_dict["user_id"][i].item(),
                     interaction_dict["item_id"][i][j],
                     interaction_dict["item_value"][i][j]))

        create_dataset_recbole(args, data)

        dataset = "eta" if args["folder"].startswith(
            "SyntheticDataset") else "youtube"
        dataset = dataset + "_retrain".format(eta)

        config_temp["eval_step"] = 99
        config_temp["dataset"] = dataset
        config_temp["data_path"] = path + folder + "recbole/" + dataset
        config_temp["eval_args"] = {
            "split": {"RS": [1.0, 0.0, 0.0]},
            "order": "RO",
            "mode": "full",
            "group_by": "user"
        }

        save_model = False

    if save_model:
        if exists(model_checkpoint_folder):
            files_to_delete = os.listdir(model_checkpoint_folder)
            for f in files_to_delete:
                if os.path.isfile(model_checkpoint_folder + f):
                    os.remove(model_checkpoint_folder + f)
        else:
            os.makedirs(model_checkpoint_folder)

    model, trainer, data, _ = get_model_structure_and_trainer(
        config=config, logger=logger, args=args, users=users, utils_dicts=utils_dicts,
        mean_slant_users=mean_slant_users, items_labels=items_labels,
        num_users=num_users, num_items=num_items, non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
        saving_path=saving_path)

    train_data, valid_data, test_data = data

    if model_name == "UserKNN":
        _, score = trainer.fit(train_data, valid_data, saved=True)

        print("\n Evaluation on test:")
        print(
            trainer.evaluate(
                test_data,
                load_best_model=False,
                show_progress=False))

    else:

        _, score = trainer.fit(
            train_data, valid_data, saved=save_model,
        )

        if save_model:
            print("SCORE Validation", score)

    return model
