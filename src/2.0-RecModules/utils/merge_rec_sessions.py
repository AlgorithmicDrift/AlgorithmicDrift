import re
import pandas as pd
from os.path import isfile, join
from os import listdir
from utils.data_utils import get_dataset_name_and_paths
import sys
import os

# add 1.0-RecVAE folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def merge_rec_sessions(
        path,
        folder,
        eta,
        strategy,
        factor,
        sub_strategy,
        retrain,
        model="RecVAE"):
    retrain = True if retrain == "1" else False

    args = {
        "path": path,
        "folder": folder,
        "strategy": strategy,
        "eta": eta,
        "factor": factor,
        "sub_strategy": sub_strategy,
    }

    _, _, rec_sessions_path = get_dataset_name_and_paths(args)

    rec_sessions_path = rec_sessions_path + model + "/sessions/"

    # Remember to check paths
    rec_sessions_name = "rec_sessions"

    if retrain:
        rec_sessions_name = rec_sessions_name + "_retrain"

    print("SAVING SESSIONS IN:", rec_sessions_path + rec_sessions_name)

    rec_session_df = None
    regex = rec_sessions_name + "_\\d+_\\d+.tsv"

    for rec_session_file in listdir(rec_sessions_path):
        rec_session_file_path = join(rec_sessions_path, rec_session_file)

        if re.search(regex, rec_session_file) is not None:
            if rec_session_df is None:
                rec_session_df = pd.read_csv(
                    rec_session_file_path, delimiter="\t")
            else:
                rec_session_df = rec_session_df.append(
                    pd.read_csv(rec_session_file_path, sep="\t"))
            # removing the previous rec sessions file
            if os.path.exists(rec_session_file_path):
                os.remove(rec_session_file_path)

    if rec_session_df is not None:
        rec_session_df.to_csv(
            join(
                rec_sessions_path,
                rec_sessions_name +
                ".tsv"),
            sep="\t",
            index=False)
