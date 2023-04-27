import os
from os.path import exists


def get_dataset_name_and_paths(args):
    path = args["path"]
    folder = args["folder"]
    strategy = args["strategy"]
    eta = args["eta"]
    factor = args["factor"]
    sub_strategy = args["sub_strategy"]

    dataset_path = ""
    dataset_name = ""

    if folder.startswith("SyntheticDataset"):

        dataset_name = "Histories_eta_" + eta
        dataset_path = path + folder + "Eta_" + eta + "/"
        saving_path = dataset_path

        if strategy == "Sparsity":
            dataset_path += "{}/{}/".format(strategy, factor)

        if strategy == "No_strategy" or strategy == "No_feedbackloop_no_strategy" or strategy == "Organic":
            saving_path += "{}/".format(strategy)

        elif strategy.startswith("Inprocessing") or strategy == "Sparsity":
            saving_path += "{}/{}/".format(strategy, factor)

        elif strategy == "Preprocessing" or strategy == "Postprocessing":
            saving_path += "{}/{}/{}/".format(strategy, factor, sub_strategy)

    return dataset_path, dataset_name, saving_path


def create_folders(dataset_path, args):
    model = args["model"]
    topk = args["topk"]
    retrain = args["retrain"]

    base_path = dataset_path + model + "/"

    model_checkpoint_folder = base_path + "model_checkpoint/"

    graphs_folder = base_path + "graphs/topk_" + str(topk) + "/"

    sessions_folder = base_path + "sessions/"

    if retrain:
        graphs_folder = graphs_folder[:-1] + "_retrain/"

    if not exists(model_checkpoint_folder):
        os.makedirs(model_checkpoint_folder)

    if not exists(graphs_folder):
        os.makedirs(graphs_folder)

    if not exists(sessions_folder):
        os.makedirs(sessions_folder)


def get_parsed_args(argv):

    path = ""
    folder = ""
    model = "RecVAE"
    # training, evaluation, generation, merge_rec_sessions, recbole_dataset
    module = "evaluation"
    eta = ""
    strategy = "No_strategy"
    factor = ""
    sub_strategy = ""
    topk = 10
    retrain = False
    user_count_start = 0
    user_count_end = 500  # num_users
    gpu_id = "0"

    if len(argv) > 1:

        if argv[2].startswith("SyntheticDataset"):
            _, path, folder, model, module, eta, strategy, factor, sub_strategy, topk, \
                retrain, user_count_start, user_count_end, gpu_id = argv

            topk = int(topk)
            user_count_start = int(user_count_start)
            user_count_end = int(user_count_end)

            retrain = True if retrain == "1" else False

            # Remember to check paths
            print(
                path,
                folder,
                model,
                module,
                eta,
                strategy,
                factor,
                sub_strategy,
                topk,
                retrain,
                user_count_start,
                user_count_end,
                gpu_id)

    keys = [
        "path",
        "folder",
        "model",
        "module",
        "eta",
        "strategy",
        "factor",
        "sub_strategy",
        "topk",
        "retrain",
        "user_count_start",
        "user_count_end",
        "gpu_id"]
    values = [path, folder, model, module, eta, strategy, factor, sub_strategy, topk,
              retrain, user_count_start, user_count_end, gpu_id]

    args = dict(zip(keys, values))

    return args
