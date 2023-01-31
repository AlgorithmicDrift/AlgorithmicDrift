import csv
import os
from os.path import exists
import pandas as pd


def get_dataset(dataset_path):

    videos_labels_dict = {}
    videos_slants_dict = {}  # real video id
    reverse_users_dict = {}
    reverse_videos_dict = {}

    users_dict = {}
    videos_dict = {}

    uir_dataset = []
    history_dataset = {}

    sess = []

    users_count = 0
    videos_count = 0

    prev_user_id = None

    with open(dataset_path, "r") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for data in reader:

            user_id = int(data["User"])
            label = data["Label"]
            video_id = int(data["Video"])
            slant = data["Slant"]
            orientation = data["Orientation"]

            if user_id not in users_dict:
                users_dict[user_id] = users_count
                reverse_users_dict[users_count] = user_id
                users_count += 1

            if video_id not in videos_dict:
                videos_dict[video_id] = videos_count
                reverse_videos_dict[videos_count] = video_id
                videos_count += 1

            uir_dataset.append(
                (users_dict[user_id],
                 videos_dict[video_id],
                 1,
                 orientation,
                 label))

            videos_labels_dict[videos_dict[video_id]] = label

            videos_slants_dict[video_id] = slant

            if prev_user_id is None or prev_user_id == users_dict[user_id]:
                sess.append(videos_dict[video_id])

            else:
                history_dataset[prev_user_id] = sess.copy()
                sess.clear()
                sess.append(videos_dict[video_id])

            prev_user_id = users_dict[user_id]

    history_dataset[prev_user_id] = sess.copy()

    utils_dicts = (
        videos_labels_dict,
        videos_slants_dict,
        reverse_users_dict,
        reverse_videos_dict)

    return history_dataset, uir_dataset, utils_dicts


def create_dataset_recbole(args, uir_dataset, rewiring_temp=False):
    path = args['path']
    folder = args['folder']
    eta = args['eta']
    retrain = args['retrain']
    factor = args['factor']
    strategy = args["strategy"]
    sub_strategy = args['sub_strategy']

    final_path = path + folder + "/recbole/"

    base_path = "eta_{}".format(eta)

    if strategy.startswith("Inprocessing") or strategy == "Sparsity":
        base_path += "_{}".format(factor)

    elif strategy == "Preprocessing" or strategy == "Postprocessing":
        base_path += "_{}_{}".format(factor, sub_strategy)

    if retrain:
        base_path += "_retrain"

    if rewiring_temp:
        base_path += "_rewiring_temp"

    final_path = final_path + base_path + "/"
    inter_name = base_path + ".inter"

    if not exists(final_path):
        os.makedirs(final_path)

    inter_col_names = ['user_id:token', 'item_id:token', 'rating:float']

    inter_path = final_path + inter_name

    df = pd.DataFrame(uir_dataset, columns=inter_col_names)
    df.to_csv(inter_path, index=False, sep='\t')
