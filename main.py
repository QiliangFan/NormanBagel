import argparse
import os
from typing import Tuple
from glob import glob
from multiprocessing import Pool

import yaml

# path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, "configs")


def load_hyper_param():
    hyperparam_path = os.path.join(CONFIG_ROOT, "hyper_params.yaml")
    hyperparam = yaml.load(hyperparam_path, Loader=yaml.FullLoader)
    return hyperparam


def load_global_config():
    global_config_path = os.path.join(CONFIG_ROOT, "global_config.yaml")
    global_config = yaml.load(global_config_path, Loader=yaml.FullLoader)
    return global_config


def work(train_files: Tuple[str, str], test_files: Tuple[str, str]):
    pass


def main():
    # data_root 不推荐由程序来创建
    data_root = global_config.DATA_ROOT
    assert os.path.exists(
        data_root), f"data root must exists, but {data_root} is not found..."
    train_root = os.path.join(data_root, "train")
    test_root = os.path.join(data_root, "test")
    assert os.path.exists(train_root) and os.path.exists(
        test_root), f"{train_root} and {test_root} must exist all !"

    study_train_files = glob(os.path.join(train_root, "**", "219", "**", "*.csv"), recursive=True)
    control_train_files = glob(os.path.join(train_root, "**", "220", "**", "*.csv"), recursive=True)
    study_test_files = glob(os.path.join(test_root, "**", "219", "**","*.csv"), recursive=True)
    control_test_files = glob(os.path.join(test_root, "**", "220", "**", "*.csv"), recursive=True)

    # 训练\测试数据对照组和实验组内容对应
    study_train_files = list(filter(lambda file: file.replace("219", "220") in control_train_files, study_train_files))
    control_train_files = [file.replace("219", "220") for file in study_train_files]
    study_test_files = list(filter(lambda file: file.replace("219", "220") in control_test_files, study_test_files))
    control_test_files = [file.replace("219", "220") for file in study_test_files]

    # 训练数据和测试数据对照组\实验组相互对应
    study_test_files = list(filter(lambda file: file.replace(test_root, train_root) in study_train_files, study_test_files))
    study_train_files = [file.replace(test_root, train_root) for file in study_test_files]
    control_test_files = [file.replace("219", "220") for file in study_test_files]
    control_train_files = [file.replace("219", "220") for file in study_train_files]

    test_files = list(zip(study_test_files, control_test_files))
    train_files = list(zip(study_train_files, control_train_files))



if __name__ == "__main__":
    hyperparam = load_hyper_param()
    global_config = load_global_config()

    main()
