from .spot import SPOT
from typing import Sequence, Tuple
import numpy as np


def _smooth(seq: np.ndarray):
    result = np.zeros_like(seq)
    for i, s in enumerate(seq):
        if s > 0:
            result[i-2:i+3] = 1
    return result


def run_spot(init_data, test_data, mad_filter: Sequence[int], q=1e-4, level=0.9) -> Tuple[np.ndarray, np.ndarray]:
    spot = SPOT(q)
    spot.fit(init_data=init_data, data=test_data)
    spot.initialize(level=level)
    result = spot.run()
    pred_label = np.zeros_like(test_data)
    thresholds = result["thresholds"]

    # detect delay : 测试检测延迟
    detect_idx = -1

    for i, (data, th, mad) in enumerate(zip(test_data, thresholds, mad_filter)):
        if data > th and mad > 0:
            if detect_idx == -1:
                detect_idx = i
            pred_label[i-2:i+3] = 1

    pred_label = _smooth(pred_label)
    return pred_label, np.asarray(thresholds), detect_idx
