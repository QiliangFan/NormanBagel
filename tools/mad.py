#!/usr/bin/env python3
"""
MAD: 用于过滤变化比较平滑的曲线
"""

import os
from typing import List, Union
import yaml

import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
THRESHOLD_FILE = os.path.join(PROJECT_PATH, "configs", "mad_threshold.yaml")

def mad(data: np.ndarray, name: str, mad_window: int):
    threshold_config = yaml.load(open(THRESHOLD_FILE, "r"), Loader=yaml.FullLoader)
    if name in threshold_config:
        threshold = threshold_config[name]
    else:
        print(f"\033[34m WARNING! \033[0m {name}'s threshold has not been configured...")
        threshold = threshold_config["default"]

    # if 1, can be set with the anomaly label
    sensitivity = np.zeros_like(data)
    for i, d in enumerate(data):
        if i - mad_window >= 0 and i + mad_window < len(data):
            before_window = data[i-mad_window:i]
            after_window = data[i+1: i+1+mad_window]
            diff = np.abs(np.median(before_window)-np.median(after_window))
            if diff > threshold:
                sensitivity[i] = 1
            else:
                sensitivity[i] = 0
        else:
            sensitivity[i] = 0
    return sensitivity
