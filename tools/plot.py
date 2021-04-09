#!/usr/bin/env python3
from typing import Sequence, Tuple

from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from bagel.data import KPI
import matplotlib.dates as mdates
import numpy as np
import os

matplotlib.use("agg")


def integrate_plot(study_test_kpi: KPI,
                   control_test_kpi: KPI,
                   anomaly_score: np.ndarray,
                   x_mean: np.ndarray,
                   x_std: np.ndarray,
                   pred_label,
                   threshold,
                   name,
                   svc,
                   save_path: str=None):
    """
    图比较多时不适合加上用英文图例`
    """
    fig_num = 3
    fig, ax = plt.subplots(fig_num, 1, figsize=(20, 10))
    plt.tight_layout(pad=5.5, h_pad=1.5)
    fig.suptitle(f"{svc}-{name}\n")

    # time axis
    ts = study_test_kpi.timestamps
    dates = [datetime.fromtimestamp(t) for t in ts]
    for i in range(fig_num):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d %H"))
        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # raw kpi
    pred_anomaly = np.copy(study_test_kpi.raw_values)
    pred_anomaly[np.where(anomaly_score == 0)] = np.inf
    ax[0].set_title("raw KPI")
    ax[0].plot(dates, study_test_kpi.raw_values, label="Study raw data")
    ax[0].plot(dates, control_test_kpi.raw_values, label="Control raw data")
    ax[0].plot(dates, pred_anomaly, label="Predict data")
    ax[0].legend()

    # anomaly score
    ax[1].set_title("Anomaly score")
    ax[1].plot(dates, anomaly_score, label="Anomaly score")
    ax[1].plot(dates, threshold, label="Threshold")
    ax[1].legend()

    # expectation
    ax[2].set_title("Expectation")
    ax[2].plot(dates, x_mean, label=r"Expectation$\mu$")
    ax[2].plot(dates, x_mean - 3 * x_std, label=r"Lower bound $\mu-3\sigma$")
    ax[2].plot(dates, x_mean + 3 * x_std, label=r"Upper bound $\mu+3\sigma$")
    ax[2].legend()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
