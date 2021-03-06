import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import re
import sys
import traceback
from glob import glob
from multiprocessing import Pool
from typing import List, Sequence, Tuple, final

import numpy as np
import yaml

import bagel
from preprocess import make_label
from spot import run_spot
from tools import mad
from tools.plot import integrate_plot

# path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, "configs").replace("\\", "/")
PARAM_SAVE = os.path.join(PROJECT_PATH, "variables").replace("\\", "/")
PLOT_FLAG = False


def load_hyper_param():
    hyperparam_path = os.path.join(CONFIG_ROOT, "hyper_params.yaml")
    hyperparam_path = hyperparam_path.replace("\\", "/")
    with open(hyperparam_path, "r") as fp:
        hyperparam = yaml.load(fp, Loader=yaml.FullLoader)
    return hyperparam


def load_global_config():
    global_config_path = os.path.join(CONFIG_ROOT, "global_config.yaml")
    global_config_path = global_config_path.replace("\\", "/")
    with open(global_config_path, "r") as fp:
        global_config = yaml.load(fp, Loader=yaml.FullLoader)
    return global_config


def filter_between_train_test(path: str, file_list: Sequence[str], train_tmp: List[str]):
    path = "/".join(path.split("/")[-2:])
    for _file in file_list:
        file = "/".join(_file.split("/")[-2:])
        if path == file:
            train_tmp.append(_file)
            return True
    return False

def work(train_files: Tuple[str, str], test_files: Tuple[str, str], hyperparam: dict, fault_list: List[dict]):
    # bagel hyperparams
    bagel_window_size = hyperparam["bagel"]["window_size"]
    time_feature = hyperparam["bagel"]["time_feature"]
    epochs = hyperparam["bagel"]["epochs"]

    study_train_file, control_train_file = train_files
    study_test_file, control_test_file = test_files

    model = bagel.Bagel(window_size=bagel_window_size,
                        time_feature=time_feature)
    # study group
    study_train_kpi = bagel.utils.load_kpi(study_train_file)
    study_test_kpi = bagel.utils.load_kpi(study_test_file)

    # load model param
    study_sign = "_".join(study_train_file.split(os.path.sep)[2:]).replace("\\", "/")
    study_model_save_path = os.path.join(
        PROJECT_PATH, "variables", study_sign).replace("\\", "/")
    if os.path.exists(os.path.join(PROJECT_PATH, "variables", study_sign + ".index").replace("\\", "/")):
        model.load(study_model_save_path)
    else:
        model.fit(study_train_kpi, epochs=epochs, verbose=0)
        model.save(study_model_save_path)
    try:
        _, _, _, pred_data = model.predict_one(study_test_kpi)

    except:
        pass

def main():
    # data_root ???????????????????????????
    data_root = global_config["DATA_ROOT"]
    fault_list = global_config["fault_injection"]

    assert os.path.exists(
        data_root), f"data root must exists, but {data_root} is not found..."
    train_root = os.path.join(data_root, "train").replace("\\", "/")
    test_root = os.path.join(data_root, "test").replace("\\", "/")
    # input_root = os.path.join(data_root, "input")
    input_root = "/home/sharespace/fanqiliang/istio"
    assert os.path.exists(train_root) and os.path.exists(
        test_root), f"{train_root} and {test_root} must exist all !"

    # make_label(global_config, input_root, test_root)
    # exit(0)
    with Pool(processes=None) as pool:
        final_test_files = []
        final_train_files = []
        for case in os.listdir(test_root):
            if case.startswith("exclude"):
                continue 
            study_train_files = glob(os.path.join(
                train_root, "**", "219", "**", "*.csv").replace("\\", "/"), recursive=True)
            study_train_files = [_.replace("\\", "/") for _ in study_train_files]
            control_train_files = glob(os.path.join(
                train_root, "**", "220", "**", "*.csv").replace("\\", "/"), recursive=True)
            control_train_files = [_.replace("\\", "/") for _ in control_train_files]
            study_test_files = glob(os.path.join(
                test_root, case, "**", "219", "**", "*.csv").replace("\\", "/"), recursive=True)
            study_test_files = [_.replace("\\", "/") for _ in study_test_files]
            control_test_files = glob(os.path.join(
                test_root, case, "**", "220", "**", "*.csv").replace("\\", "/"), recursive=True)
            control_test_files = [_.replace("\\", "/") for _ in control_test_files]

            # ??????\?????????????????????????????????????????????
            study_train_files = list(filter(lambda file: file.replace(
                "219", "220").replace("\\", "/") in control_train_files, study_train_files))
            control_train_files = [file.replace("219", "220").replace("\\", "/")
                                for file in study_train_files]
            study_test_files = list(filter(lambda file: file.replace(
                "219", "220").replace("\\", "/") in control_test_files, study_test_files))
            control_test_files = [file.replace("219", "220").replace("\\", "/")
                                for file in study_test_files]

            # ????????????????????????????????????\?????????????????????
            tmp = []
            train_tmp = []
            for f_test in study_test_files:
                if filter_between_train_test(f_test, study_train_files, train_tmp):
                    tmp.append(f_test)
                    
            study_test_files = tmp
            study_train_files = train_tmp
            control_test_files = [file.replace("219", "220")
                                for file in study_test_files]
            control_train_files = [file.replace("219", "220")
                                for file in study_train_files]

            test_files = list(zip(study_test_files, control_test_files))
            train_files = list(zip(study_train_files, control_train_files))

            final_test_files.extend(test_files)
            final_train_files.extend(train_files)

        final_test_files *= 600
        final_train_files *= 600
        final_test_files = final_test_files[:10000]
        final_train_files = final_train_files[:10000]
        pool_params = [(train, test, hyperparam, fault_list)
                    for train, test in zip(final_train_files, final_test_files)]
        import time
        start = time.time()
        # pool.starmap(work, pool_params, chunksize=len(final_test_files)//(len(final_test_files)//2))
        pool.starmap(work, pool_params, chunksize=2)
        end = time.time()
        print(f"Time cost: {end - start}s")

        pool.close()
        pool.join()
        print("Metric Num:", len(final_train_files))

if __name__ == "__main__":
    
    hyperparam = load_hyper_param()
    global_config = load_global_config()
    fault_list = global_config["fault_injection"]
    main()

