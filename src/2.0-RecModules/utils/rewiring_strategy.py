import random
import numpy as np
import pandas as pd

import os

import seaborn as sns
import matplotlib.pyplot as plt

SEED = 1
np.random.seed(SEED)


def rewiring_random(user, rewired_df, df, test_df, user_history, neutral_videos, videos_to_rewire):
    num_to_rewire = len(videos_to_rewire)

    test_user_videos = test_df[test_df["User"] == user]["Item"].unique()
    videos_to_exclude = set(test_user_videos).union(set(user_history))

    neutral_videos_never_watched = list(set(neutral_videos).difference(videos_to_exclude))

    substitute_neutral_videos = np.random.choice(neutral_videos_never_watched, num_to_rewire, replace=False)

    for i, video in enumerate(videos_to_rewire):
        neutral_substitute = substitute_neutral_videos[i]
        neutral_df_row = df[df["Item"] == neutral_substitute][["Item", "Slant", "Label"]].iloc[0].to_numpy()

        rewired_df.loc[
            (rewired_df["User"] == user) & (rewired_df["Item"] == video), ["Item", "Slant", "Label"]] = neutral_df_row

    return rewired_df


def rewiring_by_jaccard_similarity(user, rewired_df, df, test_df, user_history, videos_to_rewire, neutral_videos_users):
    neutral_video_already_used = []

    test_user_videos = test_df[test_df["User"] == user]["Item"].unique()
    videos_to_exclude = set(test_user_videos).union(set(user_history))

    # jaccard similarity in terms of common users
    for video_to_rewire in videos_to_rewire:

        # set of users who interacted with that video
        video_users = df[df["Item"] == video_to_rewire]["User"].unique()

        highest_jaccard_similarity = -1
        most_similar_neutral_video = None

        for neutral_video, neutral_video_users in neutral_videos_users.items():

            if neutral_video in neutral_video_already_used or neutral_video in videos_to_exclude:
                continue

            jaccard_similarity = round(len(set(video_users).intersection(set(neutral_video_users))) / len(
                set(video_users).union(set(neutral_video_users))), 2)

            if jaccard_similarity > highest_jaccard_similarity:
                highest_jaccard_similarity = jaccard_similarity
                most_similar_neutral_video = neutral_video

        neutral_video_already_used.append(most_similar_neutral_video)

        neutral_df_row = df[df["Item"] == most_similar_neutral_video][["Item", "Slant", "Label"]].iloc[0].to_numpy()
        rewired_df.loc[(rewired_df["User"] == user) & (rewired_df["Item"] == video_to_rewire), ["Item", "Slant",
                                                                                                "Label"]] = neutral_df_row

    return rewired_df


def rewiring_by_popularity(user, rewired_df, df, test_df, user_history, videos_to_rewire, neutral_videos_popularity):
    num_to_rewire = len(videos_to_rewire)
    neutral_videos = list(neutral_videos_popularity.keys())

    test_user_videos = test_df[test_df["User"] == user]["Item"].unique()
    videos_to_exclude = set(test_user_videos).union(set(user_history))

    neutral_videos_never_watched = list(set(neutral_videos).difference(videos_to_exclude))

    popularity = [neutral_videos_popularity[x] for x in neutral_videos_never_watched]
    popularity_distribution = [x / np.sum(popularity) for x in popularity]

    substitute_neutral_videos = np.random.choice(neutral_videos_never_watched, num_to_rewire,
                                                 replace=False, p=popularity_distribution)

    for i, video in enumerate(videos_to_rewire):
        neutral_substitute = substitute_neutral_videos[i]
        neutral_df_row = df[df["Item"] == neutral_substitute][["Item", "Slant", "Label"]].iloc[0].to_numpy()

        rewired_df.loc[
            (rewired_df["User"] == user) & (rewired_df["Item"] == video), ["Item", "Slant", "Label"]] = neutral_df_row

    return rewired_df


def get_rewired_bipartite_graph(df, test_df, rewiring_strategy, discard_percentage, neutral_videos_users,
                                neutral_videos_popularity, substitute_videos, non_rad_users, semi_rad_harmful_videos,
                                semi_rad_neutral_videos, harmful_percentage_dicts=False):
    non_rad_harmful_videos_before = {}
    non_rad_harmful_videos_after = {}

    rewired_df = df.copy()

    users_group = df.groupby("User")

    np.random.seed(1)

    for user, _ in users_group:

        # groupby on non rad users
        if user not in non_rad_users:
            continue

        user_df = users_group.get_group(user)

        user_history = user_df["Item"].unique()

        user_harmful_videos = user_df[user_df["Label"] == "harmful"]["Item"].unique()
        user_neutral_videos = user_df[user_df["Label"] == "neutral"]["Item"].unique()

        user_harmful_count = len(user_harmful_videos)

        non_rad_harmful_videos_before[user] = round((user_harmful_count / len(user_history)) * 100, 2)

        # pick x videos to rewire (where n_x is 75% of user harmful count)

        num_harmful_to_rewire = round(user_harmful_count * discard_percentage)

        # select harmful videos that are in common with semi-radicalized users
        user_harmful_videos_common_semi_rad = list(set(user_harmful_videos).intersection(set(semi_rad_harmful_videos)))
        user_harmful_videos_not_common_semi_rad = list(
            set(user_harmful_videos).difference(set(user_harmful_videos_common_semi_rad)))

        min_harmful_to_rewire = min(num_harmful_to_rewire, len(user_harmful_videos_common_semi_rad))

        harmful_videos_to_rewire = np.random.choice(user_harmful_videos_common_semi_rad, min_harmful_to_rewire,
                                                    replace=False)

        if min_harmful_to_rewire < num_harmful_to_rewire:
            deviation = num_harmful_to_rewire - min_harmful_to_rewire
            more_harmful_videos_to_rewire = np.random.choice(user_harmful_videos_not_common_semi_rad, deviation,
                                                             replace=False)

            harmful_videos_to_rewire = np.concatenate((harmful_videos_to_rewire, more_harmful_videos_to_rewire))

        # rewiring also neutral videos in common with semi-radicalized users
        user_neutral_videos_common_semi_rad = list(set(semi_rad_neutral_videos).intersection(set(user_neutral_videos)))
        num_neutral_to_rewire = round(len(user_neutral_videos_common_semi_rad) * discard_percentage)

        neutral_videos_to_rewire = np.random.choice(user_neutral_videos_common_semi_rad, num_neutral_to_rewire,
                                                    replace=False)

        if "Random" in rewiring_strategy:
            rewired_df = rewiring_random(user, rewired_df, df, test_df, user_history, substitute_videos,
                                         harmful_videos_to_rewire)

            user_history = rewired_df[rewired_df["User"] == user]["Item"].unique()
            rewired_df = rewiring_random(user, rewired_df, df, test_df, user_history, substitute_videos,
                                         neutral_videos_to_rewire)

        elif "Similarity" in rewiring_strategy:
            rewired_df = rewiring_by_jaccard_similarity(user, rewired_df, df, test_df, user_history,
                                                        harmful_videos_to_rewire,
                                                        neutral_videos_users)

            user_history = rewired_df[rewired_df["User"] == user]["Item"].unique()
            rewired_df = rewiring_by_jaccard_similarity(user, rewired_df, df, test_df, user_history,
                                                        neutral_videos_to_rewire,
                                                        neutral_videos_users)

        elif "Popularity" in rewiring_strategy:
            rewired_df = rewiring_by_popularity(user, rewired_df, df, test_df, user_history, harmful_videos_to_rewire,
                                                neutral_videos_popularity)

            user_history = rewired_df[rewired_df["User"] == user]["Item"].unique()
            rewired_df = rewiring_by_popularity(user, rewired_df, df, test_df, user_history, neutral_videos_to_rewire,
                                                neutral_videos_popularity)

        rewired_user_df = rewired_df[rewired_df["User"] == user]
        rewired_user_history = rewired_user_df["Item"].unique()

        if len(user_history) != len(rewired_user_history):
            print("Something wrong")
            break

        rewired_user_harmful_videos = rewired_user_df[rewired_user_df["Label"] == "harmful"]["Item"].unique()

        rewired_user_harmful_count = len(rewired_user_harmful_videos)
        non_rad_harmful_videos_after[user] = round((rewired_user_harmful_count / len(rewired_user_history) * 100), 2)

    if harmful_percentage_dicts:
        return rewired_df, non_rad_harmful_videos_before, non_rad_harmful_videos_after

    return rewired_df


def start_rewiring(df, test_df, rewiring_strategy, discard_percentage):
    non_rad_users = df[df["Orientation"] == "non radicalized"]["User"].unique()

    non_rad_neutral_videos = df[(df["Orientation"] == "non radicalized") & (df["Label"] == "neutral")]["Item"].unique()
    semi_rad_neutral_videos = df[(df["Orientation"] == "semi-radicalized") & (df["Label"] == "neutral")][
        "Item"].unique()
    rad_neutral_videos = df[(df["Orientation"] == "radicalized") & (df["Label"] == "neutral")]["Item"].unique()

    semi_rad_harmful_videos = df[(df["Orientation"] == "semi-radicalized") & (df["Label"] == "harmful")][
        "Item"].unique()

    only_non_rad_neutral_videos = list(set(non_rad_neutral_videos).difference(
        set(semi_rad_neutral_videos)).difference(set(rad_neutral_videos)))

    substitute_videos = only_non_rad_neutral_videos

    neutral_videos_users = {}
    neutral_videos_popularity = {}

    for neutral_video in substitute_videos:
        video_users = df[df["Item"] == neutral_video]["User"].unique()
        neutral_videos_users[neutral_video] = video_users
        neutral_videos_popularity[neutral_video] = len(video_users)

    rewired_df, h_percentages_before, h_percentages_after = get_rewired_bipartite_graph(df, test_df, rewiring_strategy,
                                                                                        discard_percentage,
                                                                                        neutral_videos_users,
                                                                                        neutral_videos_popularity,
                                                                                        substitute_videos,
                                                                                        non_rad_users=non_rad_users,
                                                                                        semi_rad_neutral_videos=semi_rad_neutral_videos,
                                                                                        semi_rad_harmful_videos=semi_rad_harmful_videos,
                                                                                        harmful_percentage_dicts=True)

    return rewired_df
