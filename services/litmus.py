import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
from litmus.litmus import Litmus
from typing import List, Union
import numpy as np
from numpy import ndarray

class LitmusAgent:
    """ There are too few parameters to be modified...
    """

    def __init__(self):
        super().__init__()

    def call(self, change_idx: int, study_data: Union[List, ndarray], control_data: Union[List, ndarray]):
        """调用litmus

        Args:
            change_idx (int): 变更点索引
            study_data (Union[List, ndarray]): 实验组数据
            control_data (Union[List, ndarray]): 对照组数据

        Raises:
            TypeError: [description]
        """
        if isinstance(study_data, List):
            study_data = np.asarray(study_data)
        if isinstance(control_data, List):
            control_data = np.asarray(control_data)
        if not isinstance(study_data, ndarray) or not isinstance(control_data, ndarray):
            raise TypeError("Unsupported Type...")
        
        # 显著性分数, litmus阈值, 实验组的预测值
        critical_score, litmus_threshold, pred_data = Litmus.run(change_idx=change_idx, study_data=study_data, control_data=control_data)

        return critical_score, litmus_threshold, pred_data