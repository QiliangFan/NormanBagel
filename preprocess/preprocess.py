import os
from glob import glob

import numpy as np
import pandas as pd


def add_header(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["timestamp", "value", "label"])
    return df


def make_label(global_config: dict, raw_root: str, test_data_root: str):
    """
    global_config: load from global_config.yaml
    raw_root: unprocessed data file
    test_data_root: path to save result and to evaluate
    """
    fault_inject_list = global_config["fault_injection"]

    raw_files = glob(os.path.join(raw_root))
