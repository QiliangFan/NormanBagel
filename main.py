import argparse
import os

import yaml

# path 
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, "configs")

def load_hyper_param():
    hyperparam_path = os.path.join(CONFIG_ROOT, "hyper_params.yaml")
    hyperparam = yaml.load(hyperparam_path, Loader=yaml.FullLoader)
    return hyperparam


def main():
    pass


if __name__ == "__main__":
    hyperparam = load_hyper_param()
    main()
