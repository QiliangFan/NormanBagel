#!/usr/bin/env python3
from typing import Sequence, Tuple

from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from bagel.data import KPI
import matplotlib.dates as mdates
import numpy as np

matplotlib.use("agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def integrate_plot(study_test_kpi: KPI,
                   control_test_kpi: KPI,
                   anomaly_score: np.ndarray,
                   x_mean: np.ndarray,
                   x_std: np.ndarray,
                   pred_label, 
                   threshold,
                   name, 
                   svc):
    """
    图比较多时不适合加上用英文图例`
    """
    fig_num = 3
    fig, ax = plt.subplots(fig_num, 1, figsize=(20, 10))
    plt.tight_layout(pad=2.5, h_pad=1.5)
    fig.suptitle(f"{svc}-{name}\n")

    # time axis
    ts = study_test_kpi.timestamps
    dates = [datetime.fromtimestamp(t) for t in ts]
    for i in range(fig_num):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d %H"))
        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # raw kpi
    ax[0].set_title("原始KPI")
    ax[0].plot(dates, study_test_kpi.raw_values, label="实验组原始数据")
    ax[0].plot(dates, control_test_kpi.raw_values, label="对照组原始数据")
    ax[0].legend()

    # anomaly score
    ax[1].set_title("异常分数")
    ax[1].plot(dates, anomaly_score, label="异常分数")
    ax[1].plot(dates, threshold, label="阈值")
    ax[1].legend()

    # expectation
    ax[2].set_title("算法预测期望")
    ax[2].plot(dates, x_mean, label=r"期望$\mu$")
    ax[2].plot(dates, x_mean - 3 * x_std, label=r"期望下限$\mu-3\sigma$")
    ax[2].plot(dates, x_mean + 3 * x_std, label=r"期望上限$\mu+3\sigma$")
    ax[2].legend()

    plt.close(fig)
