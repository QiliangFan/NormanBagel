import os
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.stats import norm

from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanAbsoluteError

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def standardize(values: np.ndarray, std, mean) -> np.ndarray:
    return (values - mean) / std

class Litmus(Model):
    epochs = 2000
    confidence = 0.9999
    THRESHOLD: float = norm.ppf(1 - (1 - confidence)/2)

    def __init__(self, out_dim, verbose=1):
        super().__init__()
        # verbose
        self.verbsoe = verbose

        # layers
        self.linear_regression: tf.keras.Model = tf.keras.Sequential([
            layers.Dense(out_dim)
        ])

    @staticmethod
    def run(change_idx: int, study_data: np.ndarray, control_data: np.ndarray, ts: np.ndarray) -> Tuple[Union[float, np.ndarray], float]:
        """
        Recommendation: 
        >>> len(study_data) == len(control_data)
        >>> True
        """
        assert len(study_data) == len(
            control_data), f"expected study_data and control_data to have the same length, but got {len(study_data)} and {len(control_data)}"

        # split data
        study_before, study_after, control_before, control_after, litmus_window_size = Litmus.split_data(change_idx, study_data, control_data)

        # data
        train_data = tf.data.Dataset.from_tensor_slices(([control_before], [study_after])).batch(litmus_window_size)
        control_before_data = tf.data.Dataset.from_tensor_slices([control_before]).batch(litmus_window_size)
        control_after_data = tf.data.Dataset.from_tensor_slices([study_after]).batch(litmus_window_size)

        # train and test
        litmus = Litmus(out_dim=litmus_window_size)
        sgd = SGD(1e-3, momentum=0.2)
        loss = MeanAbsoluteError()
        litmus.compile(sgd, loss)
        litmus.fit(train_data, epochs=2000)
        pred_control_before = litmus.predict(control_before_data)[0]  # first batch 
        pred_control_after = litmus.predict(control_after_data)[0]  # first batch

        diff_before = Litmus.compute_diff(pred_control_before, control_before)
        diff_after = Litmus.compute_diff(pred_control_after, control_after)

        _, u_yx, vx = Litmus.compute_mean_placements(diff_after, diff_before)
        _, u_xy, vy = Litmus.compute_mean_placements(diff_before, diff_after)

        critical_score = Litmus.critical_value(u_yx, u_xy, vx, vy, litmus_window_size)

        return critical_score, Litmus.THRESHOLD

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
        for d in data:
            pred_y: tf.Tensor = self(d)
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
        return (study_data[change_idx+1-litmus_window_size:change_idx+1], 
                study_data[change_idx+1:change_idx+1+litmus_window_size], 
                control_data[change_idx+1-litmus_window_size:change_idx +1], 
                control_data[change_idx+1:change_idx+1+litmus_window_size],
                litmus_window_size)

    @staticmethod
    def compute_diff(pred_data: np.ndarray, data: np.ndarray) -> np.ndarray:
        return data - pred_data

    @staticmethod
    def compute_mean_placements(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_yx_i = []
        for _x in x:
            u_yx_i.append(len(np.where(_x > y)[0]))
        u_yx_i = np.asarray(u_yx_i)
        u_yx = np.mean(u_yx_i)
        v_x = np.power(np.std(u_yx_i), 2) 
        
        return np.asarray(u_yx_i), np.asarray(u_yx), np.asarray(v_x)

    @staticmethod
    def critical_value(u_yx: np.ndarray, u_xy, v_x, v_y, window_size):
        return 0.5 * (u_yx - u_xy) * np.power(window_size, 0.5) / (np.power((u_xy*u_yx + v_x + v_y), 0.5) + 1e-6)

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return super(Litmus, self).__call__(*args, **kwargs)
