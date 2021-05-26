from typing import List, Tuple
import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
dir_path = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from bagel.data import KPI
import bagel


class AnomalyDetection:
    """
    Two interfaces are provided:
        1. Command Line Interface(CLI) (serve as a mainclass)
        2. API Call (serve as a baseclass)
    """
    PARAM_ROOT = os.path.join(dir_path, "params")

    def __init__(self):
        super().__init__()

    def run(self):
        """
        CLI interface
        ====>
        ```
        anomaly_decteion = AnomalyDetection()
        anomaly_detection.run()
        ```
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--data_dir", metavar="data_dir",
                            dest="data_dir", type=str, required=True, help="The input data.")
        parser.add_argument("-p", "--param_dir", metavar="param_dir", dest="param_dir", type=str,
                            required=False, default=None, help="The directory where checkpoint is located.")
        # bagel
        parser.add_argument("-bw", "--bagel_window_size", type=int, required=False, metavar="bagel_window_size", dest="bagel_window_size",
                            default=30, help="The window size of bagel's time series. default: `30`")
        parser.add_argument("-e", "--bagel_epochs", type=int, required=False, default=50, metavar="bagel_epochs", dest="bagel_epochs",
                            help="The number of epochs during the training step. default: `50`")
        parser.add_argument("-t", "--time_feature", type=str, required=False, default="MH", metavar="time_feature", dest="time_feature",
                            help="The dimension of time feature during the time encoding stage. `a|A|w` week; `H` one day; `I` half of the day; `M` minute; `S` second; default: `MH`")

        # median diff
        parser.add_argument("-mw", "--mad_window_size", type=int, required=False, default=5, metavar="mad_window_size", dest="mad_window_size",
                            help="The window size used for comparing the difference of median. default: `5`")

        # spot
        parser.add_argument("-n", "--spot_init_num", type=int, required=False, default=1000, metavar="spot_init_num", dest="spot_init_num",
                            help="The number of points used for initializing SPOT. default `1000`")

        config = vars(parser.parse_args())

        # load data
        train_value, train_ts, test_value, test_ts = self.load_data(
            config["data_dir"])

        self.bagel_epochs = config["bagel_epochs"]
        self.mad_window_size = config["mad_window_size"]
        self.spot_init_num = config["spot_init_num"]


        if train_value is not None and train_ts is not None:
            self.train_kpi = KPI(train_ts, train_value)
        else:
            self.train_kpi = None
        self.test_kpi = KPI(test_ts, test_value)

        self.model = bagel.Bagel(
            window_size=config["bagel_window_size"], time_feature=config["time_feature"])

    def call(self, body: dict):
        """
        `API call` interface

        Args:
            body: `data` field 
                {
                    "data": {{ body }}
                }
        --------------
        body: json
            The value is `request_body.data` 
        """
        # data
        test_value = body["test_value"]
        test_ts = body["test_ts"]

        train_value = body.get("train_value", None)
        train_ts = body.get("train_ts", None)

        # optional parameters
        bagel_window_size: int = body.get("bagel_window_size", 30)
        self.bagel_epochs: int = body.get("bagel_epochs", 50)
        time_feature: str = body.get("time_feature", "MH")
        self.mad_window_size: int = body.get("mad_window_size", 5)
        self.spot_init_num: int = body.get("spot_init_num", 1000)

        self.model = bagel.Bagel(
            window_size=bagel_window_size, time_feature=time_feature)

        if train_value is not None and train_ts is not None:
            self.train_kpi = KPI(train_ts, train_value)
        else:
            self.train_kpi = None

        self.test_kpi = KPI(test_ts, test_value)

    def train(self):
        if os.path.exists(self.PARAM_ROOT):
            print("loading model...")
            self.model.load(self.PARAM_ROOT)
        if self.train_kpi is not None:
            self.model.fit(self.train_kpi, epochs=self.bagel_epochs, verbose=0)
        self.model.save(self.PARAM_ROOT)
        print("model saved...")

    def predict(self, kpi_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """进行预测并返回结果

        Args:
            kpi_name (str): the name of kpi to be predicted

        Returns:
            Tuple[List, List, List, List, List]: 
        """
        from tools.mad import mad
        from spot.run_spot import run_spot
        anomaly_scores, x_mean, x_std = self.model.predict(self.test_kpi)

        mad_filter = mad(self.test_kpi.raw_values,
                         kpi_name, self.mad_window_size)
        pred_label, spot_threshold = run_spot(
            anomaly_scores[-self.spot_init_num:], anomaly_scores, mad_filter)

        return anomaly_scores, x_mean, x_std, pred_label, spot_threshold

    def load_data(self, data_dir: str) -> Tuple[List, List, List, List]:
        """load data from csv

        Args:
            data_dir (str): the directory where the input data located

        Raises:
            NotImplementedError: must be implemented according to specified data

        Returns:
            Tuple[List, List, List, List, List, List]: train_value, train_ts, test_value, test_ts
        """
        raise NotImplementedError(
            "This method is relative to specified data...")


if __name__ == "__main__":
    """
    Serve as the CLI interface
    """
    agent = AnomalyDetection()
    agent.run()
