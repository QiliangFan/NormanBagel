#!/usr/bin/env python3
"""
MAD: 用于过滤变化比较平滑的曲线
"""

import numpy as np


def mad(data, threshold, mad_window):
    # if 1, can be set with the anomaly label
    sensitivity = np.zeros(data)
    for i, d in enumerate(data):
        if i - mad_window >= 0 and i + mad_window < len(data):
            before_window = data[i-mad_window:i]
            after_window = data[i+1, i+1+mad_window]
            diff = np.abs(np.median(before_window)-np.median(after_window))
            if diff > threshold:
                sensitivity[i] = 1
            else:
                sensitivity[i] = 0
        else:
            sensitivity[i] = 0
    return sensitivity
