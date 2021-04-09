from .spot import SPOT
from typing import Sequence, Tuple
import numpy as np


def _smooth(seq: np.ndarray):
    for i, s in enumerate(seq):
        if s > 0:
            seq[i-2:i+3] = 1
    return seq


def run_spot(init_data, test_data, mad_filter: Sequence[int], q=1e-2, level=0.6) -> Tuple[np.ndarray, np.ndarray]:
    spot = SPOT(q)
    spot.fit(init_data=init_data, data=test_data)
    spot.initialize(level=level)
    result = spot.run()
    pred_label = np.zeros_like(test_data)
    thresholds = result["thresholds"]
    for i, (data, th, mad) in enumerate(zip(test_data, thresholds, mad_filter)):
        if data > th and mad > 0:
            pred_label[i-2:i+3] = 1

    pred_label = _smooth(pred_label)
    return pred_label, np.asarray(thresholds)
