import bagel
import pathlib
import pandas as pd
from typing import Union

from typing import Sequence

def mkdirs(*dir_list):
    for directory in dir_list:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def file_list(path: pathlib.Path) -> Sequence:
    if path.is_dir():
        return list(path.iterdir())
    return [path]


def load_kpi(file: Union[pathlib.Path, str], **kwargs) -> bagel.data.KPI:
    df = pd.read_csv(file, **kwargs)
    return bagel.data.KPI(timestamps=df.timestamp,
                          values=df.value,
                          labels=df.get('label', None),
                          name=file.stem)
