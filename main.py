import argparse
import os
from tools.plot import integrate_plot
from typing import Sequence, Tuple
from glob import glob
from multiprocessing import Pool
import bagel
from tools import mad
import yaml
from spot import run_spot
from preprocess import make_label
import sys
import re

# path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, "configs")
PARAM_SAVE = os.path.join(PROJECT_PATH, "variables")


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


def filter_between_train_test(path: str, file_list: Sequence[str]):
    path = "/".join(path.split("/")[-2:])
    for file in file_list:
        file = "/".join(file.split("/")[-2:])
        if path == file:
            return True
    return False


def work(train_files: Tuple[str, str], test_files: Tuple[str, str], hyperparam: dict):
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
        PROJECT_PATH, "varaibales", study_sign)
    if os.path.exists(os.path.join(PROJECT_PATH, "varaibales", study_sign + ".index")):
        model.load(study_model_save_path)
    else:
        model.fit(study_train_kpi, epochs=epochs)
        model.save(study_model_save_path)
    anomaly_scores, x_mean, x_std = model.predict(study_test_kpi)
    train_data_anoamly_sc, _, _ = model.predict(study_train_kpi)

    # control group
    control_train_kpi = bagel.utils.load_kpi(control_train_file)
    control_test_kpi = bagel.utils.load_kpi(control_test_file)

    # remove window_size - 1 points ahead
    anomaly_scores = anomaly_scores[bagel_window_size-1:]
    x_mean = x_mean[bagel_window_size-1:]
    x_std = x_std[bagel_window_size-1:]
    _, study_test_kpi = study_test_kpi.split_by_indices(bagel_window_size-1)
    _, control_test_kpi = control_test_kpi.split_by_indices(
        bagel_window_size-1)

    # mad
    mad_filter = mad(study_test_kpi.raw_values, name, mad_window_size)

    # spot
    pred_label, spot_threshold = run_spot(
        train_data_anoamly_sc[-spot_init_num:], anomaly_scores, mad_filter)

    # plot
    integrate_plot(study_test_kpi,
                   control_test_kpi,
                   anomaly_scores,
                   x_mean,
                   x_std,
                   pred_label,
                   spot_threshold,
                   name,
                   svc)


def main():
    # data_root 不推荐由程序来创建
    data_root = global_config["DATA_ROOT"]
    assert os.path.exists(
        data_root), f"data root must exists, but {data_root} is not found..."
    train_root = os.path.join(data_root, "train")
    test_root = os.path.join(data_root, "test")
    # input_root = os.path.join(data_root, "input")
    input_root = "/home/sharespace/fanqiliang/istio"
    assert os.path.exists(train_root) and os.path.exists(
        test_root), f"{train_root} and {test_root} must exist all !"

    make_label(global_config, input_root, test_root)

    study_train_files = glob(os.path.join(
        train_root, "**", "219", "**", "*.csv"), recursive=True)
    control_train_files = glob(os.path.join(
        train_root, "**", "220", "**", "*.csv"), recursive=True)
    study_test_files = glob(os.path.join(
        test_root, "**", "219", "**", "*.csv"), recursive=True)
    control_test_files = glob(os.path.join(
        test_root, "**", "220", "**", "*.csv"), recursive=True)

    # 训练\测试数据对照组和实验组内容对应
    study_train_files = list(filter(lambda file: file.replace(
        "219", "220") in control_train_files, study_train_files))
    control_train_files = [file.replace("219", "220")
                           for file in study_train_files]
    study_test_files = list(filter(lambda file: file.replace(
        "219", "220") in control_test_files, study_test_files))
    control_test_files = [file.replace("219", "220")
                          for file in study_test_files]

    # 训练数据和测试数据对照组\实验组相互对应
    tmp = []
    for f_test in study_test_files:
        if filter_between_train_test(f_test, study_train_files):
            tmp.append(f_test)
    study_test_files = tmp
    study_train_files = [file.replace(test_root, train_root)
                         for file in study_test_files if os.path.exists(file.replace(test_root, train_root))]
    control_test_files = [file.replace("219", "220")
                          for file in study_test_files]
    control_train_files = [file.replace("219", "220")
                           for file in study_train_files]

    test_files = list(zip(study_test_files, control_test_files))
    train_files = list(zip(study_train_files, control_train_files))

    pool_params = [(train, test, hyperparam)
                   for train, test in zip(train_files, test_files)]
    with Pool(processes=6) as pool:
        pool.starmap(work, pool_params)
        pool.close()
        pool.join()


if __name__ == "__main__":
    hyperparam = load_hyper_param()
    global_config = load_global_config()
    main()
