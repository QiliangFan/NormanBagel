import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import re
import sys
import traceback
from glob import glob
from multiprocessing import Pool
from typing import List, Sequence, Tuple

import numpy as np
import yaml

import bagel
from preprocess import make_label
from spot import run_spot
from tools import mad
from tools.plot import integrate_plot

# path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, "configs")
PARAM_SAVE = os.path.join(PROJECT_PATH, "variables")
PLOT_FLAG = True


def load_hyper_param():
    hyperparam_path = os.path.join(CONFIG_ROOT, "hyper_params.yaml")
    with open(hyperparam_path, "r") as fp:
        hyperparam = yaml.load(fp, Loader=yaml.FullLoader)
    return hyperparam


def load_global_config():
    global_config_path = os.path.join(CONFIG_ROOT, "global_config.yaml")
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

    # mad hyperparams
    mad_window_size = hyperparam["mad"]["window_size"]

    # spot hyperparams
    spot_init_num = hyperparam["spot"]["init_num"]

    study_train_file, control_train_file = train_files
    study_test_file, control_test_file = test_files

    name = os.path.splitext(os.path.basename(study_train_file))[0]
    svc = os.path.basename(os.path.dirname(study_train_file))

    model = bagel.Bagel(window_size=bagel_window_size,
                        time_feature=time_feature)

    # study group
    study_train_kpi = bagel.utils.load_kpi(study_train_file)
    study_test_kpi = bagel.utils.load_kpi(study_test_file)

    # load model param
    study_sign = "_".join(study_train_file.split(os.path.sep)[2:])
    study_model_save_path = os.path.join(
        PROJECT_PATH, "variables", study_sign)
    if os.path.exists(os.path.join(PROJECT_PATH, "variables", study_sign + ".index")):
        model.load(study_model_save_path)
    else:
        model.fit(study_train_kpi, epochs=epochs, verbose=0)
        model.save(study_model_save_path)
    try:
        anomaly_scores, x_mean, x_std = model.predict(study_test_kpi, verbose=0)
        train_data_anoamly_sc, _, _ = model.predict(study_train_kpi, verbose=0)
    except:
        print("\033[36m ????????????... \033[0m")
        return

    # control group
    control_train_kpi = bagel.utils.load_kpi(control_train_file)
    control_test_kpi = bagel.utils.load_kpi(control_test_file)

    # remove window_size - 1 points ahead
    _, study_test_kpi = study_test_kpi.split_by_indices(bagel_window_size-1)
    _, control_test_kpi = control_test_kpi.split_by_indices(
        bagel_window_size-1)

    # mad
    mad_filter = mad(study_test_kpi.raw_values, name, mad_window_size)

    # spot
    try:
        pred_label, spot_threshold, detect_idx = run_spot(
            train_data_anoamly_sc[-spot_init_num:], anomaly_scores, mad_filter)
    except:
        print("error")
        return

    # plot
    if PLOT_FLAG:
        try:
            post_sub_path = study_test_file.split(os.path.sep)[-5:]
            post_sub_path.remove("data")
            if "219" in post_sub_path: post_sub_path.remove("219")
            if "220" in post_sub_path: post_sub_path.remove("220")
            save_path = os.path.join(PROJECT_PATH, "img", os.path.sep.join(post_sub_path)).replace("csv", "png")
            split_save_path = save_path.split(os.path.sep)
            svc = split_save_path.pop(-2)
            save_path = os.path.sep.join(split_save_path)
            save_path = save_path.replace(".png", f"_{svc}.png")
            study_test_ts = study_test_kpi.timestamps
            change_ts = None
            for fault in fault_list:
                if fault["start"] >= np.min(study_test_ts) and fault["start"] <= np.max(study_test_ts):
                    change_ts = fault["start"] 
                    break
            change_idx = np.where(control_test_kpi.timestamps >= change_ts)[0][0]
            if detect_idx != -1:
                deta = detect_idx - change_idx
                print(f"{deta}")
                if deta < 0: deta = 0
                if deta > 30: return
                return deta
            else:
                print("no")
        except:
            print("\033[36m ????????????... \033[0m")
            return


def main():
    # data_root ???????????????????????????
    data_root = global_config["DATA_ROOT"]
    fault_list = global_config["fault_injection"]

    assert os.path.exists(
        data_root), f"data root must exists, but {data_root} is not found..."
    train_root = os.path.join(data_root, "train")
    test_root = os.path.join(data_root, "test")
    # input_root = os.path.join(data_root, "input")
    input_root = "/home/sharespace/fanqiliang/istio"
    assert os.path.exists(train_root) and os.path.exists(
        test_root), f"{train_root} and {test_root} must exist all !"

    num = 0
    make_label(global_config, input_root, test_root)
    # exit(0)
    with Pool(processes=None) as pool:
        final_test_files = []
        final_train_files = []
        for case in os.listdir(test_root):
            if case.startswith("exclude"):
                continue 
            print(f"CASE: {case}")
            study_train_files = glob(os.path.join(
                train_root, "**", "219", "**", "*.csv"), recursive=True)
            control_train_files = glob(os.path.join(
                train_root, "**", "220", "**", "*.csv"), recursive=True)
            study_test_files = glob(os.path.join(
                test_root, case, "**", "219", "**", "*.csv"), recursive=True)
            control_test_files = glob(os.path.join(
                test_root, case, "**", "220", "**", "*.csv"), recursive=True)

            # ??????\?????????????????????????????????????????????
            study_train_files = list(filter(lambda file: file.replace(
                "219", "220") in control_train_files, study_train_files))
            control_train_files = [file.replace("219", "220")
                                for file in study_train_files]
            study_test_files = list(filter(lambda file: file.replace(
                "219", "220") in control_test_files, study_test_files))
            control_test_files = [file.replace("219", "220")
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
            num += len(test_files)
            final_test_files.extend(test_files)
            final_train_files.extend(train_files)

            num += len(test_files)
        pool_params = [(train, test, hyperparam, fault_list)
                    for train, test in zip(final_train_files, final_test_files)]
        res = pool.starmap(work, pool_params)
        res = [item for item in res if item is not None]
        print(res)
        with open("output.txt", "a") as fp:
            for r in res:
                print(r, file=fp)
        pool.close()
        pool.join()
        print(num)

if __name__ == "__main__":
    hyperparam = load_hyper_param()
    global_config = load_global_config()
    fault_list = global_config["fault_injection"]

    main()
