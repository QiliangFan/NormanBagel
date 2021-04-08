import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_header(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["timestamp", "value", "label"])
    return df


def make_label(global_config: dict, raw_root: str, test_data_root: str) -> None:
    """
    global_config: load from global_config.yaml
    raw_root: unprocessed data file
    test_data_root: path to save result and to evaluate
    """
    fault_inject_list = global_config["fault_injection"]

    raw_files: List[str] = glob(os.path.join(raw_root))
    dst_files: List[str] = [file.replace(raw_root, test_data_root) for file in raw_files]

    for src, dst in tqdm(zip(raw_files, dst_files)):
        raw_df: pd.DataFrame = pd.read_csv(src, header=None)
        df_with_header = add_header(raw_df.values)
        for fault in fault_inject_list:
            start = fault["start"]
            end = fault["end"]
            df_with_header["label"][(df_with_header["timestamp"] >= start) & (df_with_header["timestamp"] <= end)] = 1
            del start, end

        df_with_header.to_csv(dst, index=False)
        del df_with_header, raw_df
        