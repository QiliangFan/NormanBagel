import os
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import scipy.stats
from scipy.stats import norm

from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def standardize(values: np.ndarray, std, mean) -> np.ndarray:
    return (values - mean) / std


class Litmus(Model):
    epochs = 2000
    confidence = 0.9999
    THRESHOLD = norm.ppf(1 - (1 - confidence)/2)

    def __init__(self, out_dim, verbose=1):
        super().__init__()
        # verbose
        self.verbsoe = verbose

        # layers
        self.linear_regression: tf.keras.Model = tf.keras.Sequential([
            layers.Dense(out_dim)
        ])

    @staticmethod
    def run(change_idx: int, study_data: np.ndarray, control_data: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """
        Recommendation: 
        >>> len(study_data) == len(control_data)
        >>> True
        """
        assert len(study_data) == len(
            control_data), f"expected study_data and control_data to have the same length, but got {len(study_data)} and {len(control_data)}"

        # split data
        study_before, study_after = Litmus.split(change_idx, study_data)
        control_before, control_after = Litmus.split(change_idx, study_data)

        #

    def call(self, inputs, **kwargs) -> tf.Tensor:
        return self.linear_regression(inputs)

    def fit(self, data, epochs=1000):
        for ep in range(epochs):
            for x, y in data:
                with tf.GradientTape() as tape:
                    pred_y = self(x)
                    loss = self.loss(y, pred_y)
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables))

    def predict(self, data) -> np.ndarray:
        pred_y: tf.Tensor = self(data)
        return pred_y.numpy()

    @staticmethod
    def split_data(change_idx: int, study_data: np.ndarray, control_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        if not 0 < change_idx < len(study_data) or not 0 < change_idx < len(control_data):
            raise IndexError()
        # compute minimum window size
        study_before_window = len(study_data[:change_idx+1])
        study_after_window = len(study_data[change_idx+1:])

        control_before_window = len(control_data[:change_idx+1])
        control_after_window = len(control_data[change_idx+1:])

        litmus_window_size: int = min(
            [study_before_window, study_after_window, control_before_window, control_after_window])
        return study_data[change_idx+1-litmus_window_size:change_idx+1], study_data[change_idx+1:change_idx+1+litmus_window_size],
        control_data[change_idx+1-litmus_window_size:change_idx +
                     1], control_data[change_idx+1:change_idx+1+litmus_window_size],
        litmus_window_size

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return super(Litmus, self).__call__(*args, **kwargs)
