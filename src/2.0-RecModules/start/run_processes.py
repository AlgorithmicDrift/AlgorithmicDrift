import sys
import os

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

print(parent)

from os.path import exists
from joblib import Parallel, delayed
from utils.merge_rec_sessions import merge_rec_sessions
import subprocess


def handle_processes(cmd, thread):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Started process {} with cmd = {}".format(thread, cmd))

    for line in p.stdout:
        line_str = line.decode("utf-8")
        print("Process{}".format(thread), line_str)

    p.wait()
    print(cmd, "Return code", p.returncode)

    f = open("log_processes/err_{}.txt".format(thread), "w")
    for line in p.stderr:
        f.write("{}\n".format(line.decode("utf-8")))
    f.close()

    print("Finished handle process {}".format(cmd[4]))


def run_processes(
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
        indices_call):
    num_calls = len(eta_args)

    program_to_call = 'start/handle_modules.py'

    if not exists('log_processes'):
        os.makedirs('log_processes')

    if folder.startswith('SyntheticDataset'):

        # Create recbole datasets for each eta_args
        Parallel(n_jobs=num_calls,
                 prefer='threads')(delayed(handle_processes)(["python",
                                                              program_to_call,
                                                              path,
                                                              folder,
                                                              models_args[i],
                                                              'recbole_dataset',
                                                              eta_args[i],
                                                              strategy,
                                                              factors_args[i],
                                                              sub_strategies_args[i],
                                                              topk_args[i],
                                                              retrain_args[i],
                                                              user_count_start_args,
                                                              user_count_end_args],
                                                             i) for i in range(num_calls))

        # Generate graphs/ training / evaluation for each eta_args
        if module == 'generation':
            for index in range(len(indices_call)):
                Parallel(n_jobs=len(indices_call[index]), prefer='threads')(
                    delayed(handle_processes)(["python", program_to_call, path, folder, models_args[i], module,
                                               eta_args[i], strategy, factors_args[i], sub_strategies_args[i], topk_args[i],
                                               retrain_args[i], user_count_start_args, user_count_end_args], i)
                    for i in indices_call[index])

            # Merge rec sessions for each eta_args
            if strategy != "Organic":
                Parallel(
                    n_jobs=num_calls,
                    prefer='threads')(
                    delayed(merge_rec_sessions)(
                        path,
                        folder,
                        eta_args[i],
                        strategy,
                        factors_args[i],
                        sub_strategies_args[i],
                        retrain_args[i],
                        model=models_args[i]) for i in range(num_calls))

        elif module == 'training' or module == 'evaluation':
            for i in range(num_calls):
                handle_processes(["python",
                                  program_to_call,
                                  path,
                                  folder,
                                  models_args[i],
                                  module,
                                  eta_args[i],
                                  strategy,
                                  factors_args[i],
                                  sub_strategies_args[i],
                                  topk_args[i],
                                  retrain_args[i],
                                  user_count_start_args,
                                  user_count_end_args],
                                 0)



    print("Finished all")
