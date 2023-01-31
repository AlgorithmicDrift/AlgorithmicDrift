import pandas as pd
import numpy as np

def build_harmful_weights(
        dataset,
        items_labels,
        num_users,
        num_items,
        user_idx=None,
        constant=False):

    df = pd.DataFrame(dataset, columns=["User", "Item", "Label"])

    df = df.sort_values("User")

    if user_idx is not None and not constant:
        df = df[df["User"] == user_idx]


    mean_slant_users = df[["User", "Label"]].groupby("User").mean().to_numpy()

    alpha_users = 1 - mean_slant_users

    if constant:
        alpha_users = np.repeat(np.mean(alpha_users), num_users)
        harmful_weights = np.repeat(
            alpha_users, len(items_labels)).reshape(
            (num_users, num_items + 1)) * np.array(items_labels)
    else:
        harmful_weights = np.repeat(
            alpha_users, len(items_labels)).reshape(
            (num_users, num_items + 1)) * np.array(items_labels)

    harmful_weights = (1 - harmful_weights)

    return harmful_weights


def mitigation_obj_function(scores_probs, second_parameter, lmbd=1.0):
    results = scores_probs + (lmbd * second_parameter)
    return results


def mitigation_reranking(
        scores,
        dataset,
        items_labels,
        num_users,
        num_items,
        post_processing_mitigation,
        lmbd,
        reverse_videos_dict=None,
        user_idx=None,
        T_tensor=None):
    second_parameter = 0.0

    if post_processing_mitigation == "Harmful_dynamic":
        second_parameter = build_harmful_weights(
            dataset, items_labels, num_users, num_items, user_idx, False)
    elif post_processing_mitigation == "Harmful_static":
        second_parameter = build_harmful_weights(
            dataset, items_labels, num_users, num_items, user_idx, True)

    scores = np.where(scores == -np.inf, 0, np.where(scores < 0, 0, scores))

    scores_probs = scores / np.sum(scores, axis=1, keepdims=True)

    obj_function_values = mitigation_obj_function(
        scores_probs, second_parameter, lmbd)

    return obj_function_values
