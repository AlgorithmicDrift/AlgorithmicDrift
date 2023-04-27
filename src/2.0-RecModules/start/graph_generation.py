import sys
import os
import warnings

# adding 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import mkl
mkl.set_num_threads(16)

from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
import scipy
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
# from scipy.stats import multivariate_normal

from recbole.data.utils import *
from recbole.config import Config
from recbole.utils import init_logger, init_seed
from recbole.model.context_aware_recommender import *
from recbole.trainer import *

from utils.model_utils import load_model
from utils.postprocessing_strategy import mitigation_reranking


def normalize_T(T, B):
    T /= B
    new_matrix = normalize(T, norm="l1", axis=1)
    new_matrix = np.around(new_matrix, decimals=2)

    return new_matrix


def retrieve_parameters(T, reverse_videos_dict):
    indexes = T.nonzero()

    edges_index = []
    edges_weight = []
    nodes = []

    for i, j in zip(indexes[0], indexes[1]):

        reindexed_i = reverse_videos_dict[i]
        reindexed_j = reverse_videos_dict[j]

        edges_index.append((reindexed_i, reindexed_j))
        edges_weight.append(T[i, j])

        if reindexed_i not in nodes:
            nodes.append(reindexed_i)
        if reindexed_j not in nodes:
            nodes.append(reindexed_j)

    return nodes, edges_index, edges_weight


def create_graph_tsv(
        nodes,
        edges_index,
        edges_weight,
        filename,
        transient_nodes):
    f = open(filename + "_edge.tsv", "w")
    f.write("Source\tTarget\tWeight\n")
    for i in range(len(edges_index)):
        f.write(
            "{}\t{}\t{}\n".format(
                edges_index[i][0],
                edges_index[i][1],
                edges_weight[i]))
    f.close()

    f = open(filename + "_node.tsv", "w")
    f.write("Id\tLabel\tCategory\n")
    for i in range(len(nodes)):
        category = "Neutral"
        if nodes[i] in transient_nodes:
            category = "Harmful"

        f.write("{}\t{}\t{}\n".format(nodes[i], nodes[i], category))

    f.close()
    print("saved", filename)


def init_T(sess, num_items):
    n = num_items + 1
    T = lil_matrix((n, n), dtype=np.float64)

    for i in range(len(sess) - 1):
        T[sess[i], sess[i + 1]] = 1

    return T.tocsr()


def create_T_tensor(
        history_dataset,
        user_count_start=0,
        user_count_end=100,
        num_items=0):
    T_tensor = []

    count = 0

    for user, history in list(history_dataset.items())[
                         user_count_start:user_count_end]:
        T = init_T(history, num_items)
        T_tensor.append(T)

        if count % 100 == 0:
            print("Done", count + user_count_start)

        count += 1

    return copy.deepcopy(T_tensor)


def nullify_history_scores(temp_histories, scores):
    for user, history in enumerate(temp_histories):
        scores[user, history] = -1


def save_items_exposure(
        items_exposure,
        items_exposure_path,
        retrain,
        transient_nodes,
        reverse_videos_dict,
        reverse_users_dict):
    if retrain:
        items_exposure_filename = items_exposure_path + "items_exposure_retrain.tsv"
    else:
        items_exposure_filename = items_exposure_path + "items_exposure.tsv"

    f = open(items_exposure_filename, "w")
    f.write("Item\tExposure\tLabel\tUserId\n")

    for i in range(len(items_exposure)):

        for j in range(len(items_exposure[i])):
            item = reverse_videos_dict[j]

            category = "Neutral"
            if item in transient_nodes:
                category = "Harmful"

            f.write(
                "{}\t{}\t{}\t{}\n".format(
                    item,
                    items_exposure[i][j],
                    category,
                    reverse_users_dict[i]))

    f.close()
    print("saved", items_exposure_filename)


def save_sessions(
        sessions_to_save,
        folder,
        filename,
        user_count_start,
        user_count_end,
        retrain):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if retrain:
        filename = filename + \
                   "_retrain_{}_{}.tsv".format(user_count_start, user_count_end)
    else:
        filename = filename + \
                   "_{}_{}.tsv".format(user_count_start, user_count_end)

    sessions_df = pd.DataFrame(
        sessions_to_save, columns=[
            "User", "Session", "B"])

    sessions_df.to_csv(folder + filename, sep="\t", index=False)


def simulate_organic_model_with_alpha(X, Y, noise_cov, histories, n_trials=30, shrinkage_alpha=None, verbosity=1):
    """
    Simulate organic model with alpha that controls shrinkage level.
    """
    n_users, n_dim = X.shape
    # print(n_users)
    assert Y.shape[1] == n_dim
    n_movies = Y.shape[0]
    assert noise_cov.shape == (n_dim, n_dim)
    noise_mean = np.zeros(n_dim)
    user_choices_per_trial = np.zeros((n_users, n_trials))

    if shrinkage_alpha is not None:
        assert shrinkage_alpha > 0 and shrinkage_alpha <= 1
        print('Running simulation with shrinkage alpha =', shrinkage_alpha)

    rng = np.random.default_rng()#.multivariate_normal()
    # distribution = MultivariateNormal(torch.FloatTensor(noise_mean), noise_cov)
    # distribution = multivariate_normal(noise_mean, noise_cov, allow_singular=True)

    for t in range(n_trials):

        if verbosity > 0 and t % verbosity == 0:
            print('trial', t)

        # noisy_t = time.time()
        noisy_movies = Y + rng.multivariate_normal(noise_mean, noise_cov, (n_users, n_movies))
        # distribution.sample(torch.Size([n_users, n_movies])).numpy()
        # distribution.rvs((n_users, n_movies))

        # rng.multivariate_normal(noise_mean, noise_cov, (n_users, n_movies))
        # print("Noisy time", time.time() - noisy_t)

        # for_t = time.time()
        for u in range(n_users):  # need to go user by user bc each user has their own noisy sample of movies

            if shrinkage_alpha is not None:
                noisy_mean = np.mean(noisy_movies[u], axis=0)
                noisy_movies[u] = ((1 - shrinkage_alpha) * noisy_movies[u]) + (shrinkage_alpha * noisy_mean)

            noisy_dists = cdist([X[u]], noisy_movies[u])
            assert noisy_dists.shape == (1, n_movies)

            noisy_dists[0][histories[u]] = 10000 # nullify # max(noisy_dists[0]) + 1
            # if u == 0:
            #     print(histories[0])
            user_choice = np.argmin(noisy_dists[0])  # index of movie chosen by user
            user_choices_per_trial[u, t] = user_choice
            histories[u].append(user_choice)
        # print("Time for", time.time() - for_t)
        # break


    return user_choices_per_trial


# Organic Simulation
def generate_organic_graphs(T_tensor, history_dataset, name, dataset_path, B, d, items_labels,
                            reverse_users_dict, reverse_videos_dict, transient_nodes,
                            saving_path, graphs_folder, args):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.simplefilter(
        'ignore',
        category=scipy.sparse.SparseEfficiencyWarning)

    saving_path += "a_0.4/"
    graphs_folder = saving_path + "graphs/"

    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    user_count_start = args["user_count_start"]
    user_count_end = args["user_count_end"]

    users_embeddings_fn = dataset_path + "/Users_embeddings.npy"
    items_embeddings_fn = dataset_path + "/Items_embeddings.npy"

    users_embeddings = np.load(users_embeddings_fn, allow_pickle=True)
    items_embeddings = np.load(items_embeddings_fn, allow_pickle=True)

    print(users_embeddings.shape, items_embeddings.shape)
    X = np.stack(users_embeddings[:, 1])
    Y = np.stack(items_embeddings[:, 1])

    noise_cov = np.cov(Y, rowvar=False) * 0.5
    # torch.cov(torch.FloatTensor(Y).T) * 0.5

    # noise_cov = torch.mm(noise_cov, noise_cov.t())
    # noise_cov.add_(torch.eye(noise_cov.shape[0]))

    shrinkage_alpha = 0.4

    org_alpha2results = {}

    num_users = len(X)

    ubd_array = np.zeros((num_users, B, d))

    histories = np.array(list(history_dataset.values())[
                           user_count_start:user_count_end])

    for b in range(B):
        print('B =', b)
        start = time.time()
        histories_copy = copy.deepcopy(histories)
        choices = simulate_organic_model_with_alpha(X, Y, noise_cov, histories_copy,
                                                    shrinkage_alpha=shrinkage_alpha, verbosity=20, n_trials=d)

        print("End", b, "in", time.time() - start)
        # break
        org_alpha2results[b] = choices

    for b, users_choices in org_alpha2results.items():
        for i in range(len(users_choices)):
            user_choices = users_choices[i]
            curr_item = history_dataset[i][-1]
            for j, item_index in enumerate(user_choices):  # delta loop
                item_index = int(item_index)
                original_item = int(items_embeddings[item_index][0])
                reindexed_item = list(reverse_videos_dict.keys())[list(reverse_videos_dict.values()).index(original_item)]
                T_tensor[i][curr_item, reindexed_item] += 1
                curr_item = reindexed_item

                ubd_array[i][b][j] = int(items_labels[curr_item])

    np.save(saving_path + "ubd", ubd_array)

    for i in range(len(T_tensor)):
        T_tensor[i] = normalize_T(T_tensor[i], B)

        nodes, edges, weights = retrieve_parameters(
            T_tensor[i], reverse_videos_dict)

        filename = name + "_" + str(reverse_users_dict[i + user_count_start])
        path_file = graphs_folder + filename
        # print(path_file)
        create_graph_tsv(nodes, edges, weights, path_file, transient_nodes)


# Rec-guided Simulation
def generate_graphs(
        T_tensor,
        history_dataset,
        name,
        B,
        d,
        topk=10,
        num_items=0,
        num_users=100,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        non_rad_users=None,
        semi_rad_users=None,
        saving_path="",
        reverse_users_dict=None,
        reverse_videos_dict=None,
        model_checkpoint_folder="",
        config=None,
        logger=None,
        transient_nodes=None,
        graphs_folder="",
        args=None,
        utils_dicts=None):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.simplefilter(
        'ignore',
        category=scipy.sparse.SparseEfficiencyWarning)

    user_count_start = args["user_count_start"]
    user_count_end = args["user_count_end"]

    retrain = args["retrain"]
    model_name = args["model"]

    # true_util_path = args["path"] + args["folder"] + \
    #                  "Eta_" + args["eta"] + "/True_util_eta_{}.tsv".format(args["eta"])
    # true_util = pd.read_csv(true_util_path, sep="\t").to_numpy()

    strategy = args["strategy"]

    post_processing_mitigation = None
    if strategy == "Postprocessing":
        post_processing_mitigation = args["sub_strategy"]

    model_users = np.array(users)[user_count_start:user_count_end] + 1
    model_items = np.array(list(history_dataset.values())[
                           user_count_start:user_count_end])
    model_items = [np.array(y) + 1 for y in model_items]

    items_exposure = np.zeros((num_users, num_items))

    # tuple array (user, b_iter, delta_iter, value)
    # --> value = 0 if user at simulation b_iter at step delta_iter has interacted with a neutral item, 1 otherwise
    ubd_array = np.zeros((num_users, B, d))

    sessions_folder = saving_path + model_name + "/sessions/"
    model_folder_path = saving_path + model_name + "/"

    harmful_nodes = np.where(np.array(items_labels) == 1)[0]

    saving_users = [reverse_users_dict[u]
                    for u in users[user_count_start:user_count_end]]
    saving_histories = []

    for h in list(history_dataset.values())[user_count_start:user_count_end]:
        saving_histories.append([reverse_videos_dict[x] for x in h])

    final_sessions = list(zip(saving_users, saving_histories,
                              np.zeros(len(saving_histories), dtype=int)))

    original_model, _, _, _, _, _ = load_model(
        model_checkpoint_folder=model_checkpoint_folder,
        config=config, logger=logger, utils_dicts=utils_dicts, num_items=num_items,
        non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
        args=args)
    # we have to convert scores into rank and use "inverse log" probs to sample
    position_probs = [1 / np.log(k + 1) for k in range(1, topk + 1)]
    position_probs /= np.sum(position_probs)

    utilities = []  # final "accuracies" (mean of axis=1) per user

    for iter_b in range(B):

        start_time = time.time()

        print(iter_b, "iteration of B")

        temp_histories = copy.deepcopy(model_items)

        sessions = []
        for s in list(temp_histories):
            sessions.append([reverse_videos_dict[x - 1] for x in s])

        temp_item_values = np.array([np.ones(len(x)) for x in temp_histories])

        interaction_dict = {
            "user_id": torch.LongTensor(model_users),
            "item_id": temp_histories,
            "item_value": temp_item_values
        }

        if post_processing_mitigation is not None:
            users_temp = np.concatenate([np.repeat(model_users[i], len(
                temp_histories[i])) for i in range(len(model_users))])
            items_temp = np.concatenate(temp_histories)

            labels_temp = np.where(np.isin(items_temp, harmful_nodes), 1, 0)

            uil_array = list(zip(users_temp, items_temp, labels_temp))

        # if exists(model_checkpoint_folder + "retrain"):
        #    shutil.rmtree(model_checkpoint_folder + "retrain")

        if retrain or iter_b == 0:
            model = copy.deepcopy(original_model)

        partial_utilities = np.zeros(len(users))

        for j in range(d):
            # start_d = time.time()
            interactions = Interaction(interaction_dict)

            if retrain and j % 25 == 0 and j > 0:
                model = train_model(
                    args,
                    model_checkpoint_folder=model_checkpoint_folder,
                    interaction_dict=interaction_dict,
                    config=config,
                    mean_slant_users=mean_slant_users,
                    items_labels=items_labels)

            if strategy == "No_feedbackloop_no_strategy":
                results = model.full_sort_predict(interactions)
            else:
                results = model.predict_for_graphs(interactions, user_count_start)

            scores = results.view(-1, num_items + 1).detach().cpu().numpy()
            #print("AFTER")
            #print(len(np.nonzero(scores[126])[0]))
            scores[:, 0] = -np.inf  # set scores of [pad] to -inf

            nullify_history_scores(temp_histories, scores)
            #print("AFTER NULLIFY")
            #print(len(np.nonzero(scores[126])[0]))

            if post_processing_mitigation is not None:
                lmbd = float(args["factor"].split("_")[1])
                scores = mitigation_reranking(
                    scores,
                    uil_array,
                    items_labels,
                    num_users,
                    num_items,
                    post_processing_mitigation,
                    lmbd,
                    reverse_videos_dict=reverse_videos_dict,
                    T_tensor=T_tensor)

            scores = torch.FloatTensor(scores)

            if model_name != "Random":
                topk_scores, topk_recommendations = torch.topk(scores, topk)
            else:
                topk_recommendations = []
                topk_scores = []
                for _ in range(num_users):
                    topk_recommendations.append(
                        torch.tensor(np.random.choice(range(1, num_items + 1), replace=False, size=topk)))

                # topk_recommendations = torch.tensor(np.random.choice(range(1, num_items + 1), replace=False, size=(num_users, topk)))

            temp_temp_item_values = []
            temp_temp_histories = []

            for i in range(len(topk_recommendations)):

                items_exposure[i][topk_recommendations[i].detach().cpu().numpy() - 1] += 1

                scores_probs = np.array(topk_scores[i])

                if (scores_probs == 0.).all():
                    print("T", j)
                    if i in non_rad_users:
                        print("non rad")
                    elif i in semi_rad_users:
                        print("semi rad")
                    else:
                        print("rad")
                    print("User", i)
                    print(scores_probs)
                    print("SCORES PROBS EQUAL 0")
                    exit(0)

                scores_probs /= scores_probs.sum()

                linear_combination_probs = 0.5*position_probs + 0.5*scores_probs
                linear_combination_probs /= np.sum(linear_combination_probs)

                item_sampled = np.random.choice(
                    topk_recommendations[i].detach().cpu().numpy(), p=linear_combination_probs)

                # partial_utilities[i] += true_util[i][reverse_videos_dict[item_sampled - 1]]

                # ubd_array.append((i, iter_b, j, int(items_labels[item_sampled - 1])))  # 0 if neutral, 1 otherwise
                ubd_array[i][iter_b][j] = int(items_labels[item_sampled - 1])

                T_tensor[i][temp_histories[i][-1] - 1, item_sampled - 1] += 1

                sessions[i].append(reverse_videos_dict[item_sampled - 1])

                temp_temp_histories.append(
                    np.append(temp_histories[i], item_sampled))
                temp_temp_item_values.append(
                    np.append(temp_item_values[i], 1.))

                if post_processing_mitigation is not None:
                    label = 1 if (item_sampled in harmful_nodes) else 0
                    uil_array.append((i + 1, item_sampled, label))

            temp_histories = np.array(temp_temp_histories)
            temp_item_values = np.array(temp_temp_item_values)

            interaction_dict["item_id"] = temp_histories
            interaction_dict["item_value"] = temp_item_values

            # print("iteration {}, time = {}".format(j, time.time() - start_d))

        final_sessions.extend(
            list(zip(saving_users, sessions, np.repeat(iter_b + 1, len(sessions)))))

        print("Time for an iteration of B:", time.time() - start_time)

    save_sessions(
        final_sessions,
        sessions_folder,
        "rec_sessions",
        user_count_start,
        user_count_end,
        retrain)

    mean_items_exposure = items_exposure / B

    save_items_exposure(
        mean_items_exposure,
        model_folder_path,
        retrain,
        transient_nodes,
        reverse_videos_dict,
        reverse_users_dict)

    # ubd_df = pd.DataFrame(ubd_array, columns=["User", "B", "Delta", "HarmfulInteraction"])
    # ubd_df.to_csv(model_folder_path + "ubd.csv", index=False)
    np.save(model_folder_path + "ubd", ubd_array)

    for i in range(len(T_tensor)):
        T_tensor[i] = normalize_T(T_tensor[i], B)

        nodes, edges, weights = retrieve_parameters(
            T_tensor[i], reverse_videos_dict)

        filename = name + "_" + str(reverse_users_dict[i + user_count_start])
        path_file = graphs_folder + filename
        # print(path_file)
        create_graph_tsv(nodes, edges, weights, path_file, transient_nodes)
