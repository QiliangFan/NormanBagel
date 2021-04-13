#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Sequence, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from bagel.data import KPI
from litmus import Litmus
import traceback

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
                   save_path: str = None,
                   change_ts: int = None):
    # Litmus
    try:
        if len(np.where(pred_label > 0)[0]) > 0:
            change_idx = np.where(change_ts > study_test_kpi.timestamps)[0][-1]

            critical_score, litmus_threshold = Litmus.run(change_idx=change_idx,
                                                        study_data=study_test_kpi.raw_values,
                                                        control_data=control_test_kpi.raw_values,
                                                        ts=study_test_kpi.timestamps)
        else:
            critical_score, litmus_threshold = None, None
    except Exception as e:
        traceback.print_exc()
        critical_score, litmus_threshold = None, None

    """
    图比较多时不适合加上用英文图例`
    """
    fig_num = 3
    (fig, ax) = plt.subplots(fig_num, 1, figsize=(20, 10))
    plt.tight_layout(pad=5.5, h_pad=1.5)
    if critical_score is not None and litmus_threshold is not None:
        fig.suptitle(f"{svc}-{name}\n critical score: {critical_score}, confidence interva: [ -{litmus_threshold}, {litmus_threshold}]")
    else:
        fig.suptitle(f"{svc}-{name}\n")

    # time axis
    ts = study_test_kpi.timestamps
    dates = [datetime.fromtimestamp(t) for t in ts]
    for i in range(fig_num):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d %H"))
        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # raw kpi
    pred_anomaly = np.copy(study_test_kpi.raw_values)
    pred_anomaly[np.where(pred_label == 0)] = np.inf
    if change_ts:
        ax[0].axvline(datetime.fromtimestamp(change_ts), ymin=0.02,
                      ymax=0.98, linestyle="--", color="gold")
    ax[0].set_title("raw KPI")
    ax[0].plot(dates, study_test_kpi.raw_values,
               label="Study raw data", color="mediumslateblue")
    ax[0].plot(dates, control_test_kpi.raw_values,
               label="Control raw data", color="lightsteelblue")
    ax[0].plot(dates, pred_anomaly, label="Predict data", color="lightcoral")
    ax[0].legend()

    # anomaly score
    ax[1].set_title("Anomaly score")
    ax[1].plot(dates, anomaly_score, label="Anomaly score", color="hotpink")
    ax[1].plot(dates, threshold, label="Threshold", linestyle="--", color="c")
    ax[1].legend()

    # expectation
    ax[2].set_title("Expectation")
    ax[2].plot(dates, x_mean, label=r"Expectation $\mu$", color="deepskyblue")
    ax[2].plot(dates, x_mean - 3 * x_std,
               label=r"Lower bound $\mu-3\sigma$", color="lightcyan")
    ax[2].plot(dates, x_mean + 3 * x_std,
               label=r"Upper bound $\mu+3\sigma$", color="lightcyan")
    ax[2].fill_between(dates, x_mean - 3 * x_std, x_mean +
                       3 * x_std, color="lightcyan", alpha=0.8)
    ax[2].legend()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
